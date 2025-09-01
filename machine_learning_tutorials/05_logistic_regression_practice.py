"""
로지스틱 회귀 실습
Titanic 데이터셋을 활용한 이진 분류 실습

이 실습에서는 다음을 학습합니다:
1. Titanic 데이터셋 로딩 및 탐색
2. 데이터 전처리 및 특성 엔지니어링
3. 로지스틱 회귀 모델 구축 및 훈련
4. 분류 성능 평가 및 해석
5. 파라미터 해석 및 특성 중요도 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, roc_auc_score, accuracy_score,
                           precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("로지스틱 회귀 실습: Titanic 생존 예측")
print("=" * 60)

# 1. 데이터셋 로딩 및 기본 정보 확인
print("\n1. 데이터셋 로딩 및 탐색")
print("-" * 40)

# Titanic 데이터셋 생성 (실제 환경에서는 seaborn.load_dataset('titanic') 사용)
def create_titanic_dataset():
    """Titanic 데이터셋 생성"""
    np.random.seed(42)
    n_samples = 891
    
    # 기본 특성 생성
    pclass = np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55])
    sex = np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35])
    age = np.random.normal(29.7, 14.5, n_samples)
    age = np.clip(age, 0.42, 80)  # 나이 범위 제한
    sibsp = np.random.poisson(0.5, n_samples)
    parch = np.random.poisson(0.4, n_samples)
    fare = np.random.lognormal(3.2, 1.3, n_samples)
    embarked = np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
    
    # 생존 확률 계산 (실제 패턴 반영)
    survival_prob = 0.1  # 기본 생존 확률
    
    # 성별 영향 (여성이 생존율 높음)
    survival_prob += np.where(sex == 'female', 0.6, -0.1)
    
    # 클래스 영향 (1등급이 생존율 높음)
    survival_prob += np.where(pclass == 1, 0.3, 
                             np.where(pclass == 2, 0.1, -0.2))
    
    # 나이 영향 (어린이와 중년이 생존율 높음)
    survival_prob += np.where(age < 16, 0.2, 0)
    
    # 가족 수 영향
    family_size = sibsp + parch
    survival_prob += np.where((family_size >= 1) & (family_size <= 3), 0.1, -0.1)
    
    # 요금 영향
    survival_prob += np.where(fare > np.percentile(fare, 75), 0.2, 0)
    
    # 확률을 0-1 범위로 제한
    survival_prob = np.clip(survival_prob, 0, 1)
    
    # 생존 여부 결정
    survived = np.random.binomial(1, survival_prob, n_samples)
    
    # 결측치 추가
    age_missing = np.random.choice(n_samples, int(0.2 * n_samples), replace=False)
    age[age_missing] = np.nan
    
    embarked_missing = np.random.choice(n_samples, 2, replace=False)
    embarked[embarked_missing] = np.nan
    
    # 데이터프레임 생성
    data = pd.DataFrame({
        'survived': survived,
        'pclass': pclass,
        'sex': sex,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'fare': fare,
        'embarked': embarked
    })
    
    return data

# 데이터 로딩
titanic = create_titanic_dataset()

print(f"데이터셋 크기: {titanic.shape}")
print(f"특성 수: {titanic.shape[1] - 1}")
print(f"샘플 수: {titanic.shape[0]}")

print("\n데이터 기본 정보:")
print(titanic.info())

print("\n타겟 변수 분포:")
print(titanic['survived'].value_counts())
print(f"생존율: {titanic['survived'].mean():.3f}")

# 2. 탐색적 데이터 분석
print("\n2. 탐색적 데이터 분석")
print("-" * 40)

# 기술 통계
print("수치형 변수 기술 통계:")
print(titanic.describe())

print("\n범주형 변수 분포:")
categorical_cols = ['pclass', 'sex', 'embarked']
for col in categorical_cols:
    print(f"\n{col}:")
    print(titanic[col].value_counts())

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Titanic Dataset Exploratory Data Analysis', fontsize=16)

# 생존율 by 성별
survival_by_sex = titanic.groupby('sex')['survived'].mean()
axes[0, 0].bar(survival_by_sex.index, survival_by_sex.values)
axes[0, 0].set_title('Survival Rate by Gender')
axes[0, 0].set_ylabel('Survival Rate')

# 생존율 by 클래스
survival_by_class = titanic.groupby('pclass')['survived'].mean()
axes[0, 1].bar(survival_by_class.index, survival_by_class.values)
axes[0, 1].set_title('Survival Rate by Class')
axes[0, 1].set_ylabel('Survival Rate')

# 나이 분포
axes[0, 2].hist(titanic['age'].dropna(), bins=30, alpha=0.7)
axes[0, 2].set_title('Age Distribution')
axes[0, 2].set_xlabel('Age')

# 생존자 vs 비생존자 나이 분포
axes[1, 0].hist(titanic[titanic['survived']==0]['age'].dropna(), 
                bins=30, alpha=0.7, label='Not Survived')
axes[1, 0].hist(titanic[titanic['survived']==1]['age'].dropna(), 
                bins=30, alpha=0.7, label='Survived')
axes[1, 0].set_title('Age Distribution by Survival')
axes[1, 0].legend()

# 요금 분포
axes[1, 1].hist(titanic['fare'], bins=30, alpha=0.7)
axes[1, 1].set_title('Fare Distribution')
axes[1, 1].set_xlabel('Fare')

# 가족 크기 vs 생존율
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
survival_by_family = titanic.groupby('family_size')['survived'].mean()
axes[1, 2].bar(survival_by_family.index, survival_by_family.values)
axes[1, 2].set_title('Survival Rate by Family Size')
axes[1, 2].set_xlabel('Family Size')

plt.tight_layout()
plt.show()

# 3. 데이터 전처리 및 특성 엔지니어링
print("\n3. 데이터 전처리 및 특성 엔지니어링")
print("-" * 40)

# 데이터 복사
df = titanic.copy()

print("결측치 확인:")
print(df.isnull().sum())

# 결측치 처리
print("\n결측치 처리:")

# 나이 결측치: 중앙값으로 대체
age_median = df['age'].median()
df['age'].fillna(age_median, inplace=True)
print(f"나이 결측치를 중앙값({age_median:.1f})으로 대체")

# 승선 항구 결측치: 최빈값으로 대체
embarked_mode = df['embarked'].mode()[0]
df['embarked'].fillna(embarked_mode, inplace=True)
print(f"승선 항구 결측치를 최빈값({embarked_mode})으로 대체")

# 특성 엔지니어링
print("\n특성 엔지니어링:")

# 1. 가족 크기
df['family_size'] = df['sibsp'] + df['parch'] + 1
print("가족 크기 특성 생성")

# 2. 혼자 여행 여부
df['is_alone'] = (df['family_size'] == 1).astype(int)
print("혼자 여행 여부 특성 생성")

# 3. 나이 그룹
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 60, 100], 
                        labels=['Child', 'Teen', 'Adult', 'Senior'])
print("나이 그룹 특성 생성")

# 4. 요금 그룹
df['fare_group'] = pd.qcut(df['fare'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
print("요금 그룹 특성 생성")

# 범주형 변수 인코딩
print("\n범주형 변수 인코딩:")

# 성별 인코딩
df['sex_encoded'] = LabelEncoder().fit_transform(df['sex'])
print("성별 인코딩 완료")

# 승선 항구 원-핫 인코딩
embarked_dummies = pd.get_dummies(df['embarked'], prefix='embarked')
df = pd.concat([df, embarked_dummies], axis=1)
print("승선 항구 원-핫 인코딩 완료")

# 나이 그룹 원-핫 인코딩
age_group_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
df = pd.concat([df, age_group_dummies], axis=1)
print("나이 그룹 원-핫 인코딩 완료")

# 요금 그룹 원-핫 인코딩
fare_group_dummies = pd.get_dummies(df['fare_group'], prefix='fare_group')
df = pd.concat([df, fare_group_dummies], axis=1)
print("요금 그룹 원-핫 인코딩 완료")

# 특성 선택
feature_columns = [
    'pclass', 'sex_encoded', 'age', 'sibsp', 'parch', 'fare',
    'family_size', 'is_alone',
    'embarked_C', 'embarked_Q', 'embarked_S',
    'age_group_Child', 'age_group_Teen', 'age_group_Adult', 'age_group_Senior',
    'fare_group_Low', 'fare_group_Medium', 'fare_group_High', 'fare_group_Very_High'
]

X = df[feature_columns]
y = df['survived']

print(f"\n최종 특성 수: {X.shape[1]}")
print("선택된 특성들:")
for i, col in enumerate(feature_columns, 1):
    print(f"{i:2d}. {col}")

# 4. 데이터 분할 및 스케일링
print("\n4. 데이터 분할 및 스케일링")
print("-" * 40)

# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"훈련 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")
print(f"훈련 데이터 생존율: {y_train.mean():.3f}")
print(f"테스트 데이터 생존율: {y_test.mean():.3f}")

# 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("특성 스케일링 완료")

# 5. 로지스틱 회귀 모델 구축 및 훈련
print("\n5. 로지스틱 회귀 모델 구축 및 훈련")
print("-" * 40)

# 기본 로지스틱 회귀 모델
print("기본 로지스틱 회귀 모델:")
lr_basic = LogisticRegression(random_state=42)
lr_basic.fit(X_train_scaled, y_train)

# 기본 성능 평가
y_pred_basic = lr_basic.predict(X_test_scaled)
y_pred_proba_basic = lr_basic.predict_proba(X_test_scaled)[:, 1]

print(f"기본 모델 정확도: {accuracy_score(y_test, y_pred_basic):.4f}")
print(f"기본 모델 AUC: {roc_auc_score(y_test, y_pred_proba_basic):.4f}")

# 하이퍼파라미터 튜닝
print("\n하이퍼파라미터 튜닝:")
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 교차검증 AUC: {grid_search.best_score_:.4f}")

# 최적 모델
lr_best = grid_search.best_estimator_

# 6. 모델 성능 평가
print("\n6. 모델 성능 평가")
print("-" * 40)

# 예측
y_pred = lr_best.predict(X_test_scaled)
y_pred_proba = lr_best.predict_proba(X_test_scaled)[:, 1]

# 성능 지표 계산
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print("분류 성능 지표:")
print(f"정확도 (Accuracy): {accuracy:.4f}")
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc_score:.4f}")

# 상세 분류 리포트
print("\n상세 분류 리포트:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# 혼동 행렬
print("\n혼동 행렬:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 교차 검증
cv_scores = cross_val_score(lr_best, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"\n교차 검증 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 7. 결과 시각화
print("\n7. 결과 시각화")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Logistic Regression Results', fontsize=16)

# 혼동 행렬 히트맵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'],
            ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# ROC 곡선
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend(loc="lower right")

# 특성 중요도 (회귀 계수)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': lr_best.coef_[0],
    'abs_coefficient': np.abs(lr_best.coef_[0])
}).sort_values('abs_coefficient', ascending=True)

top_features = feature_importance.tail(10)
axes[1, 0].barh(range(len(top_features)), top_features['coefficient'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'])
axes[1, 0].set_title('Top 10 Feature Coefficients')
axes[1, 0].set_xlabel('Coefficient Value')

# 예측 확률 분포
axes[1, 1].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, 
                label='Not Survived', density=True)
axes[1, 1].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, 
                label='Survived', density=True)
axes[1, 1].set_title('Predicted Probability Distribution')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# 8. 파라미터 해석 및 특성 중요도 분석
print("\n8. 파라미터 해석 및 특성 중요도 분석")
print("-" * 40)

print("회귀 계수 해석:")
print(f"절편 (Intercept): {lr_best.intercept_[0]:.4f}")

print("\n특성별 회귀 계수 (상위 10개):")
for _, row in feature_importance.tail(10).iterrows():
    coef = row['coefficient']
    feature = row['feature']
    odds_ratio = np.exp(coef)
    
    if coef > 0:
        effect = "증가"
        direction = "높아짐"
    else:
        effect = "감소"
        direction = "낮아짐"
    
    print(f"{feature:20s}: {coef:7.4f} (오즈비: {odds_ratio:.3f})")
    print(f"  → 이 특성이 {effect}하면 생존 확률이 {direction}")

# 9. 다양한 임계값에서의 성능 분석
print("\n9. 다양한 임계값에서의 성능 분석")
print("-" * 40)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("임계값별 성능:")
print("Threshold | Accuracy | Precision | Recall | F1-Score")
print("-" * 50)

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)
    f1_thresh = f1_score(y_test, y_pred_thresh)
    
    print(f"   {threshold:.1f}    |  {acc:.4f}  |   {prec:.4f}  |  {rec:.4f} |  {f1_thresh:.4f}")

# 10. 예측 예시
print("\n10. 예측 예시")
print("-" * 40)

# 몇 가지 샘플에 대한 예측
sample_indices = [0, 1, 2, 3, 4]
print("샘플 예측 결과:")

for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = y_pred[idx]
    probability = y_pred_proba[idx]
    
    print(f"\n샘플 {idx + 1}:")
    print(f"  실제 생존: {'Yes' if actual == 1 else 'No'}")
    print(f"  예측 생존: {'Yes' if predicted == 1 else 'No'}")
    print(f"  생존 확률: {probability:.3f}")
    print(f"  예측 정확: {'✓' if actual == predicted else '✗'}")

# 11. 모델 요약
print("\n11. 모델 요약")
print("-" * 40)

print("로지스틱 회귀 모델 요약:")
print(f"• 사용된 특성 수: {len(feature_columns)}")
print(f"• 최적 정규화 강도 (C): {lr_best.C}")
print(f"• 정규화 방법: {lr_best.penalty}")
print(f"• 테스트 정확도: {accuracy:.4f}")
print(f"• 테스트 AUC: {auc_score:.4f}")

print("\n주요 발견사항:")
print("• 성별이 생존에 가장 큰 영향을 미침 (여성의 생존율이 높음)")
print("• 승객 등급이 중요한 요인 (1등급 승객의 생존율이 높음)")
print("• 나이와 가족 구성도 생존에 영향을 미침")
print("• 로지스틱 회귀는 해석 가능한 분류 모델로 유용함")

print("\n실습 완료!")
print("로지스틱 회귀를 통해 Titanic 승객의 생존을 성공적으로 예측했습니다.")