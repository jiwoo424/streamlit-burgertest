import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle
import sklearn
import re
import tqdm
import numpy as np
import pandas as pd
import zipfile
import scipy
import folium
import os
import subprocess

from PIL import Image
from zipfile import ZipFile
from ALS import als_model, rest2idx, idx2rest, data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from implicit.als import AlternatingLeastSquares as ALS  

st.set_page_config(layout="wide")


project_dir = 'projects'


if not os.path.exists(project_dir):
    st.write("Cloning repository...")
    cmd = 'git clone -b master --single-branch https://github.com/Korea-sehun/projects.git'
    subprocess.check_call(cmd, shell=True)
    st.write("Repository cloned.")
else:
    st.write(f"Directory '{project_dir}' already exists. Skipping clone.")


image_dir = "projects/real_image"
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

def find_photo(menu_list, name_list, base_folder):
    for menu, name in zip(menu_list, name_list):
	    file_name = f"{menu}_{name}.jpg"
	    file_path = os.path.join(base_folder, file_name)
	    return file_path

# ''' Backend '''
franchise_burger = pd.read_csv('df_franchise.csv')
premium_burger = pd.read_csv('df_premium_final.csv')

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
    selected_burger = re.sub(r'\[.*?\]\s*', '', selected_burger_input)
    selected_burger_patty = burger_data.loc[burger_data['menu'] == selected_burger, 'patty'].values
    if len(selected_burger_patty) == 0:
        # 선택된 버거가 데이터에 없는 경우 처리
        return pd.DataFrame()
    selected_burger_patty = selected_burger_patty[0]

    final_scores = similarity_df[selected_burger].values
    recommendations_df = pd.DataFrame({
        'id': burger_data['id'].tolist(),
    'menu': burger_data['menu'].tolist(),
    'name': burger_data['name'].tolist(),
    'class': burger_data['class'].tolist(),
    'price': burger_data['price'].tolist(),
    'patty': burger_data['patty'].tolist(),
    'visitor': burger_data['visitor'].tolist(),
    'blog': burger_data['blog'].tolist(),
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
    final_recommendations = filtered_recommendations[['id', 'menu', 'name', 'price', 'score']] \
        .sort_values(by='score', ascending=False) \
        .iloc[:10]
    return final_recommendations

# ''' Frontend '''
st.title("[KUBIG 19기 추천시스템팀] 수제버거 추천시스템")
v = st.write(""" <h2> <b style="color:red"> 수제버거 </b> 추천시스템 🍔</h2>""",unsafe_allow_html=True)
st.write(""" <p> 프랜차이즈 버거로 취향 저격 <b style="color:red">수제버거</b> 찾기! </p>""",unsafe_allow_html=True)
my_expander = st.expander("Tap to Select a Burger 🍔")
selected_burger_name = my_expander.selectbox("내가 좋아하는 프랜차이즈 버거는",franchise_burger['menu_input'])
price_range = my_expander.slider("가격 범위 설정", value=[0, 42500])


if my_expander.button("Recommend"):
    st.text("Here are few Recommendations..")
    result = final_recommendation(burger_data, selected_burger_name, price_range[0], price_range[1], 0)
    menu_list = result['menu'].tolist()
    id_list = result['id'].tolist()
    name_list = result['name'].tolist()
    price_list = result['price'].tolist()
    score_list = result['score'].tolist()
    unique_names = []
    burger_image_path = []	
	
    for name in name_list:
        if name not in unique_names:
            unique_names.append(name)
        if len(unique_names) == 5:
            break  
    related = als_model.similar_items(rest2idx[unique_names[0]])
    array2list = related[0]
    number_list = array2list.tolist()
    result_list = []
    for idx in number_list:
        rest_ids = data[data['restidx'] == idx]['rest_id'].unique()
        for rest_id in rest_ids:
            if rest_id not in unique_names:
                result_list.append(rest_id)
    v = st.write("""<h2> 당신의 <b style="color:red"> 수제버거 </b> 취향은? </h2>""",unsafe_allow_html=True)
    col1,col2,col3,col4,col5=st.columns(5)
    cols=[col1,col2,col3,col4,col5]
    if not menu_list:
        st.write('<b style="color:#E50914"> Sorry, no results found! </b>', unsafe_allow_html=True)
        st.text("가격 범위를 늘려보세요 😢")
    else:
        for i in range(0,5):
			burger_image = find_photo(menu_list[i], name_list[i], base_folder)
			burger_image_path.append(burger_image)
            rank = i + 1
            with cols[i]:
                st.write(f'{rank}위')
                st.write(f' <b style="color:#E50914"> {menu_list[i]} </b>',unsafe_allow_html=True)
                try:
                    image = Image.open(burger_image_path[i])
                    st.image(image, caption=menu_list[i], use_column_width=True)
                except FileNotFoundError:
                    st.write("[no image]")
                    
                st.write("________")
                st.write(f'<b style="color:#DB4437">가게명</b>:<b> {name_list[i]}</b>',unsafe_allow_html=True)
                st.write(f'<b style="color:#DB4437">   Price  </b>: <b> {price_list[i]} <b> ',unsafe_allow_html=True)
    v = st.write(""" <h2> 방문해보면 좋을 수제버거 <b style="color:red"> 가게 </b> 추천 </h2>""",unsafe_allow_html=True)
    col1,col2,col3,col4,col5=st.columns(5)
    cols=[col1,col2,col3,col4,col5]
    for i in range(0,5):
        rank = i + 1
        with cols[i]:
            st.write(f'{rank}위')
            st.write(f' <b style="color:#E50914"> {result_list[i]} </b>',unsafe_allow_html=True)
            # st.write("#")
            st.write("________")
    lati = []
    longi = []
    for store_name in result_list:
	    matching_rows = burger_data[burger_data['name'] == store_name]
	    unique_lat_lon = matching_rows[['latitude', 'longitude']].drop_duplicates()
	    lati.extend(unique_lat_lon['latitude'].tolist())
	    longi.extend(unique_lat_lon['longitude'].tolist())
    map_data = pd.DataFrame({
    'lat': lati[0:5],
    'lon': longi[0:5],
    'name': result_list[0:5],
    })
    my_map = folium.Map(
	location=[map_data['lat'].mean(), map_data['lon'].mean()], 
    zoom_start=7)
    for index, row in map_data.iterrows():
	    folium.Marker(
		    location=[row['lat'], row['lon']],   # 값 표시 위치 (위도, 경도)
		    popup=row['name'],                   # 팝업에 가게 이름 표시
		    icon=folium.Icon(icon='info-sign')      # 기본 아이콘 사용 (옵션)
	    ).add_to(my_map)
	    
    st.components.v1.html(my_map._repr_html_(), width=800, height=600)
