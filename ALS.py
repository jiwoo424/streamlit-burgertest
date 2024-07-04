import numpy as np
import tqdm 
from implicit.als import AlternatingLeastSquares as ALS  
import pandas as pd
import scipy
from zipfile import ZipFile
import zipfile
import scipy.sparse as sparse



# ALS 협업 필터링
zip_file = 'review_data.csv.zip'
csv_file = 'review_data.csv'

# ZIP 파일 열기
with zipfile.ZipFile(zip_file, 'r') as zipf:
    with zipf.open(csv_file) as file:
        train = pd.read_csv(file)



train = train[['username','restaurant']]
train.columns = ['user_id', 'rest_id']

# 데이터 <--> 인덱스 교환 딕셔너리
user2idx = {l: i for i, l in enumerate(train['user_id'].unique())}
rest2idx = {l: i for i, l in enumerate(train['rest_id'].unique())}
idx2user = {i: l for i, l in enumerate(train['user_id'].unique())}
idx2rest = {i: l for i, l in enumerate(train['rest_id'].unique())}

# 데이터에 인덱스 추가
data = train.copy()
data['user_idx'] = data['user_id'].map(user2idx)
data['rest_idx'] = data['rest_id'].map(rest2idx)
rating = np.ones(len(data))

# 희소 행렬 생성
purchase_sparse = sparse.csr_matrix((rating, (data['user_idx'], data['rest_idx'])),
                                   shape=(len(user2idx), len(rest2idx)))

# ALS 모델 초기화 및 학습
als_model = ALS(factors=40, regularization=0.01, iterations=50)
als_model.fit(purchase_sparse, show_progress=False)

# 내보낼 변수 정의
__all__ = ['als_model', 'rest2idx', 'user2idx', 'idx2user', 'idx2rest', 'data']
