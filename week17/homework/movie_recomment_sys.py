class MovieRecommendationSystem:
    def __init__(self, ratings, movies):
        """
        初始化推荐系统
        :param ratings: 评分数据字典 {user_id: {movie_id: rating}}
        :param movies: 电影数据字典 {movie_id: {title, genres, director, actors}}
        """
        self.ratings = ratings
        self.movies = movies

    def compute_item_similarity(self):
        """
        计算物品相似度矩阵， -> 物品的协同过滤推荐
        :return: 相似度矩阵字典 {movie_id: {similar_movie_id: similarity_score}}
        """
        pass

    def item_based_recommend(self, user_id, top_n=10):
        """
        基于物品的协同过滤推荐(相似度推荐)
        :param user_id: 用户ID
        :param top_n: 返回Top-N推荐
        :return: 推荐列表 [(movie_id, score, reason), ...]
        """
        pass

    def build_user_profile(self, user_id):
        """
        构建用户画像 ->内容推荐
        :param user_id: 用户ID
        :return: 用户偏好向量
        """
        pass

    def content_based_recommend(self, user_id, top_n=10):
        """
        基于内容的推荐，用户历史评分与电影特征
        :param user_id: 用户ID
        :param top_n: 返回Top-N推荐
        :return: 推荐列表 [(movie_id, score, reason), ...]
        """
        pass

    def hybrid_recommend(self, user_id, top_n=10, cf_weight=0.6):
        """
        混合推荐
        :param user_id: 用户ID
        :param top_n: 返回Top-N推荐
        :param cf_weight: 协同过滤权重（0-1）
        :return: 推荐列表 [(movie_id, score, reason), ...]
        """
        pass