# tree-models
Let's learn from simple tree models to XGBoooooost

## \<CART: Classification & Regression Tree\>

#### 1. Classification Tree

#### 2. Regression Tree

- Regression Trees, Clearly Explained!!! :  https://www.youtube.com/watch?v=g9c66TUylZ4  
- How to Prune Regression Trees, Clearly Explained!!!: https://www.youtube.com/watch?v=D0efHEJsfHo


## \<Keywords\>

1. Pruning
2. Impurity



## \<Metrics\>

1. Entropy & Information Gain: https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8
2. Gini Impurity


### XGBoost 용어 정의

1. Output Value
2. Residual
3. Similarity Score
4. Gain
5. Gamma(γ)
6. Lambda(λ)



## Missing Value Imputation

1. XGBoost is not black magic: https://towardsdatascience.com/xgboost-is-not-black-magic-56ca013144b4
2. StatQuest: Random Forests Part 2: Missing data and clustering: https://www.youtube.com/watch?v=sQ870aTKqiM  

   -> proximity matrix: 샘플 x 샘플 matrix로 m 샘플과 n 샘플이 같은 leaf에서 끝나면 (m,n) (n,m)에 1을 더하고 이 프로세스를 전체 tree 개수에 대해 진행함.  
   
   -> 이 과정을 converge 할 때까지 반복함
3. StatQuest: Decision Trees, Part 2 - Feature Selection and Missing Data: https://www.youtube.com/watch?v=wpNl-JwwplA



## 공식링크

1. 공식 도큐먼트: https://xgboost.readthedocs.io/en/latest/index.html
2. Awesome XGBoost: https://github.com/dmlc/xgboost/tree/master/demo


## 블로그 & 튜토리얼

1. Using XGBoost in Python: https://www.datacamp.com/community/tutorials/xgboost-in-python
2. XGBoost is not black magic(feat. missing value): https://towardsdatascience.com/xgboost-is-not-black-magic-56ca013144b4 


## 코드

1. Pure Python implementation of Gradient Boosted Trees: https://github.com/dimleve/tinygbt  
  -> [최건호가 주석 달아놓은 코드](./tiny-gbt)


## 강의

1. XGBoost Regressor: https://www.youtube.com/watch?v=OtD8wVaFm6E

2. Gradient Boost Main Ideas: https://www.youtube.com/watch?v=3CC4N4z3GJc

3. Decision tree: https://www.youtube.com/watch?v=7VeUPuFGJHk
3. StatQuest: Decision Trees, Part 2 - Feature Selection and Missing Data :   https://www.youtube.com/watch?v=wpNl-JwwplA


4. Random Forests Part 1: https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
5. Random Forests Part 2: https://www.youtube.com/watch?v=nyxTdL_4Q-Q -> 나중에 다시 보자

6. Adaboost: https://www.youtube.com/watch?v=LsK-xG1cLYA  
  -> Decision Node 가 하나인 depth=1인 tree를 서로 다른 가중치로 순서대로 배열한것.  
  -> 각 tree power는 작지만 이걸 이어붙이면 나름 쓸만하다.
  
7. Gradient Boost Part 1: https://www.youtube.com/watch?v=3CC4N4z3GJc
8. Gradient Boost Part 2 - Regression: https://www.youtube.com/watch?v=2xudPOBz-vs
9. Gradient Boost Part 3 - Classification: https://www.youtube.com/watch?v=jxuNLH5dXCs
10. Gradient Boost Part 4 - Classification Details: https://www.youtube.com/watch?v=StWY5QWMXCw  
  -> 수식 파티! 예~
  
11. XGBoost Part 1: XGBoost Trees for Regression: https://www.youtube.com/watch?v=OtD8wVaFm6E
12. XGBoost Part 2: XGBoost Trees For Classification: https://www.youtube.com/watch?v=8b1JEDvenQU 
13. XGBoost Part 3: Mathematical Details: https://www.youtube.com/watch?v=ZVFeW798-2I
14. XGBoost Part 4: Crazy Cool Optimizations: https://www.youtube.com/watch?v=oRrKeUCEbq8  
-> Sketch Algorithm(?) 여기 내용들은 나중에 실제로 최적화를 해봐야 공감 및 이해가 될듯
