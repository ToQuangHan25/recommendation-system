import streamlit as st
import numpy as np
import pandas as pd

# 1. Class Linear Regression
class RidgeRegression:
    def __init__(self, lr=0.1, epochs=1000, lambda_=0.01):
        self.lr = lr
        self.epochs = epochs
        self.lambda_=lambda_
        self.w = None
        self.b = None
    def predict(self, X):
        return X.dot(self.w) + self.b
    def gradient(self, X, y):
        y_pred = self.predict(X)
        samples = X.shape[0]
        db = 2 / samples * np.sum(y_pred - y)
        dw = 2 / samples * X.T.dot(y_pred - y) + 2 * self.lambda_ * self.w
        return db, dw
    def fit(self, X, y):
        self.b, self.w = 0.0, np.zeros(X.shape[1])
        for i in range(self.epochs):
            db, dw = self.gradient(X, y)
            self.b -= self.lr * db
            self.w -= self.lr * dw

# 2. Hàm tải dữ liệu
@st.cache_data
def load_data():
    ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_cols
    movies = pd.read_csv('u.item', sep='|', encoding='latin-1', names=cols)
    genre_cols=genre_cols[1:]

    # Tính điểm cộng đồng
    movie_stats = ratings.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
    movie_stats.rename(columns={'mean': 'avg_rating', 'count': 'num_ratings'}, inplace=True)
    
    movies = pd.merge(movies, movie_stats, on='movie_id', how='left')
    movies['avg_rating'] = movies['avg_rating'].fillna(0)
    movies['num_ratings'] = movies['num_ratings'].fillna(0)
    return ratings, movies, genre_cols

# 3. Thiết kế giao diện
st.set_page_config(page_title="Recommend System", layout="wide") # Mở rộng giao diện ra toàn màn hình
st.title("🎬 Hệ Thống Gợi Ý Phim")

# Tải dữ liệu
ratings, movies, genre_cols = load_data()

# Tạo hai tab chính
tab_data, tab_new_user = st.tabs(["Thư viện phim", "Gợi ý"])

# with tab_rec:
#     # Ô nhập User ID
#     TARGET_USER = st.sidebar.number_input('Nhập User ID (từ 1 đến 943):', min_value=1, max_value=943, value=1, step=1)
#     NUM_MOVIES = st.sidebar.number_input('Số lượng phim muốn gợi ý:', min_value=1, max_value=20, value=5, step=1)
#     # Khi người dùng bấm nút
#     if st.button('Gợi ý phim'):
#         # Hiển thị hiệu ứng loading
#         with st.spinner(f'Đang phân tích sở thích của User {TARGET_USER}...'):
            
#             # 1. Lọc dữ liệu
#             user_ratings = ratings[ratings['user_id'] == TARGET_USER]
#             user_data = pd.merge(user_ratings, movies, on='movie_id')
            
#             # Kiểm tra nếu người dùng chưa xem phim nào
#             if len(user_data) == 0:
#                 st.warning("Người dùng này chưa đánh giá bộ phim nào. Hãy thử ID khác!")
#             else:
#                 # 2. Huấn luyện mô hình
#                 X_train = user_data[genre_cols].values
#                 y_train = user_data['rating'].values
                
#                 model = RidgeRegression(lr=0.1, epochs=1000, lambda_=0.01) # Hạ lr để an toàn hơn
#                 model.fit(X_train, y_train)
                
#                 # 3. Dự đoán phim chưa xem
#                 seen_movie_ids = user_ratings['movie_id'].tolist()
#                 unseen_movies = movies[~movies['movie_id'].isin(seen_movie_ids)].copy()
#                 X_test = unseen_movies[genre_cols].values
                
#                 pred = model.predict(X_test)
#                 unseen_movies['predicted_rating']=np.clip(pred, 1.0, 5.0)
#                 top_recommendations = unseen_movies.sort_values(by='predicted_rating', ascending=False).head(NUM_MOVIES)
#                 st.divider()
#                 with st.expander(f"📜 Xem lịch sử đánh giá của User {TARGET_USER} ({len(user_data)} phim)"):
#                     # Sắp xếp phim theo điểm rating từ cao xuống thấp để xem họ thích phim nào nhất
#                     user_history = user_data[['title', 'rating']].sort_values(by='rating', ascending=False)
                    
#                     # Hiển thị bảng dữ liệu gọn gàng
#                     st.dataframe(
#                         user_history,
#                         use_container_width=True,
#                         hide_index=True,
#                         column_config={
#                             "title": "Tên phim",
#                             "rating": st.column_config.NumberColumn("Đánh giá", format="%d ⭐"),
#                         }
#                     )
#                 # 4. Hiển thị kết quả ra web
#                 st.success(f"Đã tìm thấy Top {NUM_MOVIES} bộ phim phù hợp nhất với User {TARGET_USER}!")
                
#                 # In ra danh sách phim đẹp mắt
#                 for idx, row in top_recommendations.iterrows():
#                     # Tạo một khung bao quanh mỗi bộ phim
#                     with st.container():
#                         # Chia làm 2 cột: 1 cột cho tên phim, 1 cột cho nút bấm link
#                         c1, c2 = st.columns([3, 1])
            
#                     with c1:
#                         st.markdown(f"🎞️ **{row['title']}**")
#                         # Giới hạn rating trong khoảng 1-5 sao bằng np.clip
#                         final_rating = np.clip(row['predicted_rating'], 1.0, 5.0)
#                         st.write(f"⭐ Dự đoán: {final_rating:.1f} sao")
            
#                     with c2:
#                         # Kiểm tra xem có link không, nếu có thì hiện nút
#                         url = f"https://www.google.com/search?q={row['title'].replace(' ', '+')}+movie"
#                         if pd.isna(url) or url == "":
#                             st.write("Không có link")
#                         else:
#                             # Tạo nút bấm mở link trong tab mới
#                             st.link_button("Link", url)
            
#                     st.divider() # Thêm một đường gạch ngang ngăn cách giữa các phim
with tab_data:
    st.subheader(f"Khám phá toàn bộ **{len(movies)}** bộ phim")
    
    # Thêm thanh tìm kiếm phim
    search_input = st.text_input("🔍 Tìm tên phim:", "")
    
    def get_movie_genres(row):
        return ", ".join([genre for genre in genre_cols if row[genre] == 1])

    # Tạo một bản sao để không làm hỏng dữ liệu gốc
    df_display = movies.copy()
    
    # Tạo cột 'Genres' bằng cách áp dụng hàm trên cho từng hàng (axis=1)
    df_display['Genres'] = df_display.apply(get_movie_genres, axis=1)
    
    # Tạo cột link Google
    df_display['Google_Link'] = "https://www.google.com/search?q=" + df_display['title'].str.replace(' ', '+').str.replace('&', '%26')
    
    # Tạo cột 'Genres' tổng hợp
    display_cols = ['movie_id', 'title', 'Genres', 'Google_Link']
    
    # Lọc theo tìm kiếm
    if search_input:
        df_display = df_display[df_display['title'].str.contains(search_input, case=False)]
        
    # Hiển thị bảng
    st.dataframe(
        df_display[display_cols],
        width='stretch',
        hide_index=True,
        column_config={
            "movie_id": "ID",
            "title": "Tên Phim",
            "Genres": "Thể loại",
            "Google_Link": st.column_config.LinkColumn("Link phim", display_text="Link")
        }
    )
with tab_new_user:
    st.write("Chọn thể loại phim bạn yêu thích")
    
    # Chia giao diện thành 3 cột cho gọn gàng
    cols = st.columns(3)
    user_genre_preferences = []
    
    # Tạo thanh trượt (slider)
    for i, genre in enumerate(genre_cols):
        with cols[i % 3]: # Chia đều các slider vào 3 cột
            is_selected = st.checkbox(f"🎬 {genre}")
            score = 5.0 if is_selected else 0.0
            user_genre_preferences.append(score)
            
    # Nút bấm lấy gợi ý
    st.divider()
    NUM_MOVIES_NEW = st.number_input('Chọn số lượng phim:', min_value=1, max_value=20, value=5, step=1, key='new_user_k')
    
    if st.button('Gợi ý phim', key='user'):
        if sum(user_genre_preferences) == 0:
            st.warning("⚠️ Bạn hãy chọn ít nhất một thể loại phim")
        else:
            with st.spinner('Đang tính toán sở thích...'):
                X_train = np.eye(len(genre_cols))
                y_train = np.array(user_genre_preferences)
                
                model = RidgeRegression(lr=0.1, epochs=1000, lambda_=0.01)
                model.fit(X_train, y_train)
                
                # Xử lý dữ liệu
                X_all_movies = movies[genre_cols].values
                
                # Đếm xem mỗi phim có bao nhiêu thể loại (Tính tổng theo hàng ngang)
                # keepdims=True để giữ nguyên cấu trúc cột dọc để chia ma trận
                row_sums = X_all_movies.sum(axis=1, keepdims=True)
                
                # Tránh lỗi chia cho 0
                row_sums[row_sums == 0] = 1 
                
                # Chia giá trị (0 hoặc 1) cho tổng số thể loại của phim đó
                X_test = X_all_movies / row_sums
                
                # Dự đoán trên dữ liệu đã chuẩn hóa
                y_pred = model.predict(X_test)
                
                movies_new_user = movies.copy()
                movies_new_user['raw_score'] = y_pred
                
                # BỘ LỌC 1: Bỏ qua những phim ít người xem (dưới 20 lượt vote) để tránh phim rác
                valid_movies = movies_new_user[movies_new_user['num_ratings'] >= 20].copy()
                
                # BỘ LỌC 2: Xếp hạng ưu tiên những phim có độ "Khớp Gu" cao nhất
                # nếu độ khớp gu bằng nhau thì ưu tiên phim có "Điểm Cộng Đồng" cao hơn
                valid_movies['final_ranking_score'] = valid_movies['raw_score'] + valid_movies['avg_rating']
                
                # Lấy Top phim
                top_new_recs = valid_movies.sort_values(by='final_ranking_score', ascending=False).head(NUM_MOVIES_NEW)
                top_recs = top_new_recs.sort_values(by='avg_rating', ascending=False).head(NUM_MOVIES_NEW)
                # 4. Hiển thị kết quả
                st.success(f"Đã tìm thấy Top {NUM_MOVIES_NEW} bộ phim phù hợp nhất cho bạn:")
                
                for idx, row in top_recs.iterrows():
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"🎞️ **{row['title']}**")
                            st.write(f"Rating: {row['avg_rating']:.1f} ⭐")
                            m_genres = [g for g in genre_cols if row[g] == 1]
                            st.caption(f"_{', '.join(m_genres)}_")
                        with c2:
                            url = f"https://www.google.com/search?q={row['title'].replace(' ', '+').replace('&', '%26')}"
                            if pd.isna(url) or url == "":
                                st.write("Không có link")
                            else:
                                # Tạo nút bấm mở link trong tab mới
                                st.link_button("Link", url)