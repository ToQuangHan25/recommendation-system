import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

class RidgeRegression(LinearRegression):
    def __init__(self, lr=0.1, epochs=1000, lambda_=0.01):
        super().__init__(lr, epochs)
        self.lambda_=lambda_
    def loss_function(self, y_pred, y):
        return np.mean((y_pred-y)**2)+self.lambda_*np.sum(self.w**2)
    def gradient(self, X, y):
        y_pred=self.predict(X)
        samples=X.shape[0]
        db=2/samples*np.sum(y_pred-y)
        dw=2/samples*X.T.dot(y_pred-y)+2*self.lambda_*self.w
        return db, dw

print("Đang tải và xử lý dữ liệu...")
# 1. Tải dữ liệu Ratings
ratings = pd.read_csv("u.data", sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
print('Utility Matrix')
utility_matrix=ratings.pivot(index='user_id', columns='movie_id', values='rating')
print(utility_matrix.iloc[:10,:10])

total=utility_matrix.shape[0]*utility_matrix.shape[1]
rated=utility_matrix.count().sum()
sparsity=((total-rated)/total)*100
print(f'Sparsity: {sparsity}')

# 2. Tải dữ liệu Phim và 19 cột Thể loại
genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
              'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_cols

movies = pd.read_csv("u.item", sep='|', encoding='latin-1', names=cols)

# print(ratings.head(5))
print(movies.head(5))

TARGET_USER = int(input('Nhập User: '))  # Chọn User số 1 để gợi ý
k=int(input('Nhập số phim: '))
print(f"\nĐang học sở thích của User ID = {TARGET_USER}...")

# Lọc ra những phim User 1 đã xem và chấm điểm
user_ratings = ratings[ratings['user_id'] == TARGET_USER]
# print(user_ratings)
user_data = pd.merge(user_ratings, movies, on='movie_id')
# print(user_data)

# Định nghĩa X (19 thể loại phim) và y (điểm rating)
X_train = user_data[genre_cols].values
y_train = user_data['rating'].values
# print(X_train.shape, y_train.shape)
# Huấn luyện mô hình Linear Regression
model = RidgeRegression()
model.fit(X_train, y_train)

# Tìm danh sách các phim User 1 chưa xem
seen_movie_ids = user_ratings['movie_id'].tolist()
# print(seen_movie_ids)
unseen_movies = movies[~movies['movie_id'].isin(seen_movie_ids)].copy()
# print(unseen_movies)

# Lấy đặc trưng X của các phim chưa xem
X_test = unseen_movies[genre_cols].values

# Dùng mô hình tuyến tính dự đoán điểm user sẽ chấm
pred = model.predict(X_test)
unseen_movies['predicted_rating']=np.clip(pred, 1.0, 5.0)
# print(unseen_movies)

# Sắp xếp điểm dự đoán từ cao xuống thấp và lấy Top 5
top_recommendations = unseen_movies.sort_values(by='predicted_rating', ascending=False).head(k)
print(top_recommendations)

print(f"\n🎉 KẾT QUẢ GỢI Ý TOP {k} PHIM:")
for idx, row in top_recommendations.iterrows():
    print(f"- Phim: {row['title']:<40} | Dự đoán: {row['predicted_rating']:.2f} sao")