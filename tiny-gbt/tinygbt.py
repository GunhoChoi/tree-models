#!/usr/bin/python
'''
    File name: tinygbt.py
    Author: Seong-Jin Kim
    EMail: lancifollia@gmail.com
    Date created: 7/15/2018
    Reference:
        [1] T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. 2016.
        [2] G. Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. 2017.
'''

import sys
import time
try:
    # For python2
    from itertools import izip as zip
LARGE_NUMBER = sys.maxint
except ImportError:
    # For python3
    LARGE_NUMBER = sys.maxsize

import numpy as np
from sklearn.metrics import log_loss

class Dataset(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y


class TreeNode(object):
    def __init__(self):
        self.is_leaf = False
        self.left_child = None
        self.right_child = None
        # Feature to split
        self.split_feature_id = None
        # Feature value for split
        self.split_val = None
        # 이 구현체의 weight는 output value를 의미함
        self.weight = None

   
    def _calc_split_gain(self, G, H, G_l, H_l, G_r, H_r, lambd):
        """
        G: Gradient(1st order derivative)
        H: Hessian(2nd order derivative)
        l & r: left & right

        Loss reduction
        (Refer to Eq7 of Reference[1])
        """
        def calc_term(g, h):
            return np.square(g) / (h + lambd)
        return calc_term(G_l, H_l) + calc_term(G_r, H_r) - calc_term(G, H)


    def _calc_leaf_weight(self, grad, hessian, lambd):
        """
        Calculate the optimal weight of this leaf node.
        (Refer to Eq5 of Reference[1])
        """
        return -np.sum(grad) / (np.sum(hessian) + lambd)


    def build(self, instances, grad, hessian, shrinkage_rate, depth, param):
        """
        Exact Greedy Alogirithm for Split Finidng
        (Refer to Algorithm1 of Reference[1])
        """
        assert instances.shape[0] == len(grad) == len(hessian)

        if depth > param['max_depth']:
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grad, hessian, param['lambda']) * shrinkage_rate
            return

        G = np.sum(grad)
        H = np.sum(hessian)
        best_gain = 0.
        best_feature_id = None
        best_val = 0.
        best_left_instance_ids = None
        best_right_instance_ids = None

        for feature_id in range(instances.shape[1]):
            G_l, H_l = 0., 0.
            # 음 sorted_instance_ids 가 정확히 뭘까 
            # 왜 sort 하는거지?
            # 일단 argsort는 작은순서대로 index를 정렬해서 전달함
            sorted_instance_ids = instances[:,feature_id].argsort()
            
            # j는 그냥 sorted 된거 쭉 도는거 0~n
            for j in range(sorted_instance_ids.shape[0]):
                # split 지점을 작은 수치부터 높은 수치까지 돌면서 찾는데 
                # G, H 를 이미 받았으니까 해당 split 지점 기준 왼쪽 오른쪽 분리하고 
                # 거기까지 G와 H를 누적해 G_l, 이를 제외한 값을 G_r에 할당하고 이때 Gain을 계산함
                # print(instances[:,feature_id][sorted_instance_ids[j]])
                G_l += grad[sorted_instance_ids[j]]
                H_l += hessian[sorted_instance_ids[j]]
                G_r = G - G_l
                H_r = H - H_l
                current_gain = self._calc_split_gain(G, H, G_l, H_l, G_r, H_r, param['lambda'])
               
                # 이렇게 돌면서 best Gain을 주는 feature와 이때 분리 지점을 찾아 저장함
                # 근데 이러면 O(n^2) 이라 느릴듯
                # 그래서 xgboost에서는 다양한 최적화 방법을 쓰는듯하다
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_feature_id = feature_id
                    best_val = instances[sorted_instance_ids[j]][feature_id]
                    best_left_instance_ids = sorted_instance_ids[:j+1]
                    best_right_instance_ids = sorted_instance_ids[j+1:]
        
        # best GAIN이더라도 지정한 minimum gain 보다 낮으면 하위 노드 생성을 멈추고
        # leaf로 지정하고 output value(여기서 weight) 설정
        if best_gain < param['min_split_gain']:
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grad, hessian, param['lambda']) * shrinkage_rate
        
        # minimum gain보다 높으면 트리노드를 생성해 left & right child 를 생성함
        else:
            # best gain 의 feature 가 뭐였는지 저장하고
            # 최대 gain 지점을 split val에 할당함
            self.split_feature_id = best_feature_id
            self.split_val = best_val

            # 노드 생성
            # left side data만 넘김 
            # depth+1
            self.left_child = TreeNode()
            self.left_child.build(instances[best_left_instance_ids], 
                                  grad[best_left_instance_ids],
                                  hessian[best_left_instance_ids],
                                  shrinkage_rate,
                                  depth+1, param)

            # 노드 생성
            # right side data만 넘김 
            # depth+1
            self.right_child = TreeNode()
            self.right_child.build(instances[best_right_instance_ids],
                                   grad[best_right_instance_ids],
                                   hessian[best_right_instance_ids],
                                   shrinkage_rate,
                                   depth+1, param)


    # predict는 쭉쭉 타고 내려가서 leaf 노드의 weight(또는 output value) 값을 가져오는것
    def predict(self, x):
        if self.is_leaf:
            return self.weight
        else:
            if x[self.split_feature_id] <= self.split_val:
                return self.left_child.predict(x)
            else:
                return self.right_child.predict(x)


class Tree(object):
    ''' Classification and regression tree for tree ensemble '''
    def __init__(self):
        self.root = None

    def build(self, instances, grad, hessian, shrinkage_rate, param):
        assert len(instances) == len(grad) == len(hessian)
        self.root = TreeNode()
        current_depth = 0
        self.root.build(instances, grad, hessian, shrinkage_rate, current_depth, param)

    def predict(self, x):
        return self.root.predict(x)


class GBT(object):
    def __init__(self):
        self.params = {'gamma': 0.,
                       'lambda': 1.,
                       'min_split_gain': 0.1,
                       'max_depth': 5,
                       'learning_rate': 0.3,
                       }
        self.best_iteration = None
    
    def _calc_training_data_scores(self, train_set, models):
        ''' 
        여기서 score가 예측 값인듯하다 
        '''
        if len(models) == 0:
            return None
        X = train_set.X
        scores = np.zeros(len(X))
        for i in range(len(X)):
            scores[i] = self.predict(X[i], models=models)
        return scores

    def _calc_l2_gradient(self, train_set, scores):
        '''
        L2 loss를 1,2차 미분해보면 왜 hessian, grad가 아래와 같은 식으로 나오는지 확인할 수 있다
        '''
        labels = train_set.y
        hessian = np.full(len(labels), 2)
        if scores is None:
            grad = np.random.uniform(size=len(labels))
        else:
            grad = np.array([2 * (scores[i] - labels[i]) for i in range(len(labels))])
        return grad, hessian

    def _calc_gradient(self, train_set, scores):
        """For now, only L2 loss is supported"""
        return self._calc_l2_gradient(train_set, scores)

    def _calc_l2_loss(self, models, data_set):
        '''
        l2 오류 계산
        '''
        errors = []
        for x, y in zip(data_set.X, data_set.y):
            errors.append(y - self.predict(x, models))
        return np.mean(np.square(errors))

    def _calc_loss(self, models, data_set):
        """For now, only L2 loss is supported"""
        return self._calc_l2_loss(models, data_set)

    def sigmoid(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def logLikelihoodLoss(self, y_hat, y_true):
        prob = self.sigmoid(y_hat)
        grad = prob - y_true
        hess = prob * (1.0 - prob)
        return grad, hess

    def _calc_log_loss_gradient(self, train_set, scores):
        labels = train_set.y
        if scores is None:
            grad = np.random.uniform(size=len(labels))
            hessian = np.random.uniform(size=len(labels))
        else:
            grad, hessian = self.logLikelihoodLoss(scores, labels)
        return grad, hessian

    def _calc_logloss_loss(self, models, data_set):
        preds = []
        for x, y in zip(data_set.X, data_set.y):
            preds.append(self.predict(x, models))

        return log_loss(data_set.y, self.sigmoid(np.array(preds)))

    def _calc_log_loss(self, models, data_set):
        return self._calc_logloss_loss(models, data_set)

    def _build_learner(self, train_set, grad, hessian, shrinkage_rate):
        '''
        하나의 트리를 만들고 parameter에 따라 트리 생성하는 함수
        '''
        learner = Tree()
        learner.build(train_set.X, grad, hessian, shrinkage_rate, self.params)
        return learner

    def train(self, params, train_set, num_boost_round=20, valid_set=None, early_stopping_rounds=5, objective="regression"):
        self.params.update(params)
        models = []
        shrinkage_rate = 1.
        best_iteration = None
        best_val_loss = LARGE_NUMBER
        train_start_time = time.time()

        print("Training until validation scores don't improve for {} rounds."
              .format(early_stopping_rounds))

        # tree의 개수에 따라 range만큼 반복!
        for iter_cnt in range(num_boost_round):
            iter_start_time = time.time()
            # score 는 예측값
            scores = self._calc_training_data_scores(train_set, models)
            
            # 정답과 예측값을 통해 gradient 및 hessian 계산
            if objective in "regression":
                # 처음에 모델이 없을때는 scores에 None 이 리턴됨
                # gradient는 [0,1) uniform, hessian은 2
                grad, hessian = self._calc_gradient(train_set, scores)
            elif objective in "binary":
                # 처음에 모델이 없을때는 scores에 None 이 리턴됨
                # gradient과 hessian 모두 [0,1) uniform
                grad, hessian = self._calc_log_loss_gradient(train_set, scores)

            # tree 생성
            learner = self._build_learner(train_set, grad, hessian, shrinkage_rate)

            # shrinkage 지속적으로 감소하는구나
            if iter_cnt > 0:
                shrinkage_rate *= self.params['learning_rate']
                
            # list of models 에 append
            models.append(learner)

            if objective in "regression":
                train_loss = self._calc_loss(models, train_set)
                val_loss = self._calc_loss(models, valid_set) if valid_set else None
                val_loss_str = '{:.10f}'.format(val_loss) if val_loss else '-'
                print("Iter {:>3}, Train's L2: {:.10f}, Valid's L2: {}, Elapsed: {:.2f} secs"
                  .format(iter_cnt, train_loss, val_loss_str, time.time() - iter_start_time))

            elif objective in "binary":
                train_loss = self._calc_log_loss(models, train_set)
                val_loss = self._calc_log_loss(models, valid_set) if valid_set else None
                val_loss_str = '{:.10f}'.format(val_loss) if val_loss else '-'
                print("Iter {:>3}, Train's LogLoss: {:.10f}, Valid's LogLoss: {}, Elapsed: {:.2f} secs"
                      .format(iter_cnt, train_loss, val_loss_str, time.time() - iter_start_time))

            # best validation 찾기
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = iter_cnt

            # early stopping 걸기
            if iter_cnt - best_iteration >= early_stopping_rounds:
                print("Early stopping, best iteration is:")
                print("Iter {:>3}, Train's L2: {:.10f}".format(best_iteration, best_val_loss))
                break

        self.models = models
        self.best_iteration = best_iteration
        print("Training finished. Elapsed: {:.2f} secs".format(time.time() - train_start_time))

    # 예측값
    def predict(self, x, models=None, num_iteration=None):
        if models is None:
            models = self.models
        assert models is not None
        return np.sum(m.predict(x) for m in models[:num_iteration])
