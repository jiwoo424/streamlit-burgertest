import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle
import sklearn
import pyautogui
import mouseinfo
    _display = Display(os.environ['DISPLAY'])


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



st.set_page_config(layout="wide")


# ''' Backend '''
franchise_burger = pd.read_csv('df_franchise.csv')
premium_burger = pd.read_csv('df_premium.csv')

# 데이터 전처리
# 프리미엄 버거 가격 숫자형 데이터로 바꾸는 작업
# '원' 지우고 진행
premium_burger['price'] = premium_burger['price'].replace({'원': '', ',': ''}, regex=True)
# 숫자형으로 변환할 수 없는 경우 nan 처리
premium_burger['price'] = pd.to_numeric(premium_burger['price'], errors='coerce')
# price에서 결측값 있는 행 제거
premium_burger = premium_burger.dropna(subset=['price'])
# 정수형으로 바꾸기
premium_burger['price'] = premium_burger['price'].astype(int)
franchise_burger = franchise_burger.rename(columns={
    'brand': 'name',
    'name': 'menu',
})


franchise_burger['menu_input'] = '[' + franchise_burger['name'] + '] ' + franchise_burger['menu']


# 프랜차이즈에서 관련있는 행만 모으기
filtered_franchise_burger = franchise_burger[[ 'name', 'menu', 'price', 'wordlist', 'patty']]
# 프랜차이즈, 프리미엄 클래스 구분
filtered_franchise_burger['class'] = 0
premium_burger['class'] = 1
burger_data = pd.concat([filtered_franchise_burger, premium_burger], ignore_index = True)

burger_data['visitor'] = burger_data['visitor'].fillna('0')
burger_data['blog'] = burger_data['blog'].fillna('0')

burger_data['visitor'] = burger_data['visitor'].str.replace(',', '').astype(int)
burger_data['blog'] = burger_data['blog'].str.replace(',', '').astype(int)

max_visitors = burger_data['visitor'].max()
max_blogs = burger_data['blog'].max()


burger_data['wordlist'] = burger_data['wordlist'].astype(str)

tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(' '))
tfidf_matrix = tfidf_vectorizer.fit_transform(burger_data['wordlist'])

# 코사인 유사도 계산
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 코사인 유사도를 DataFrame으로 변환하여 확인
similarity_df = pd.DataFrame(cosine_sim, index=burger_data['name'], columns=burger_data['menu'])
similarity_df.head()


def final_recommendation(burger_data, selected_burger_input, min, max, popularity_min):
  receive = selected_burger_input.split()
  selected_burger = receive[1]
  selected_burger_patty = burger_data[burger_data['menu'] == selected_burger]['patty'].values[0]

  final_scores = similarity_df[selected_burger].values

  recommendations_df = pd.DataFrame({
      'id': burger_data['id'],
    'menu': burger_data['menu'],
    'name': burger_data['name'],
    'class': burger_data['class'],
    'price': burger_data['price'],
    'patty': burger_data['patty'],
    'visitor': burger_data['visitor'],
    'blog': burger_data['blog'],
    'score': final_scores
  })

  filtered_recommendations = recommendations_df[
        (recommendations_df['class'] == 1) &
        (recommendations_df['price'] >= min) &
        (recommendations_df['price'] < max) &
        ((recommendations_df['visitor'] + recommendations_df['blog']) >= popularity_min) &
        (recommendations_df['patty'].str.contains(selected_burger_patty, na = False))
  ]

  filtered_recommendations = filtered_recommendations.drop_duplicates(subset='menu')

  # 최종 추천 점수를 계산하여 상위 5개 추천 확인
  final_recommendations = filtered_recommendations[['id', 'menu', 'name', 'price', 'score']].sort_values(by='score', ascending=False).iloc[1:11]
  return final_recommendations




# ''' Frontend '''
st.title("[KUBIG 19기 추천시스템팀] 수제버거 추천시스템")
v = st.write(""" <h2> <b style="color:red"> 수제버거 </b> 추천시스템 🍔</h2>""",unsafe_allow_html=True)
st.write(""" <p> 프랜차이즈 버거로 취향 저격 <b style="color:red">수제버거</b> 찾기! </p>""",unsafe_allow_html=True)
my_expander = st.expander("Tap to Select a Burger 🍔")
selected_burger_name = my_expander.selectbox("내가 좋아하는 프랜차이즈 버거는",franchise_burger['menu_input'])
price_range = my_expander.slider("가격 범위 설정", value=[0, 60000])


if my_expander.button("Recommend"):
    st.text("Here are few Recommendations..")
    st.write("#")
    result = final_recommendation(burger_data, selected_burger_name, price_range[0], price_range[1], 0)
    menu_list = result['menu'].tolist()
    id_list = result['id'].tolist()
    name_list = result['name'].tolist()
    price_list = result['price'].tolist()
    score_list = result['score'].tolist()
    col1,col2,col3,col4,col5=st.columns(5)
    cols=[col1,col2,col3,col4,col5]
    for i in range(0,5):
            with cols[i]:
                rank = i + 1
                with cols[i]:
                    st.write(f'{rank}위')
                st.write(f' <b style="color:#E50914"> {menu_list[i]} </b>',unsafe_allow_html=True)
                # st.write("#")
                st.write("________")
                st.write(f'<b style="color:#DB4437">가게명</b>:<b> {name_list[i]}</b>',unsafe_allow_html=True)
                st.write(f'<b style="color:#DB4437">   Price  </b>: <b> {price_list[i]} <b> ',unsafe_allow_html=True)

if st.button("Reset"):
    pyautogui.hotkey("ctrl","F5")
