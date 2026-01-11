import math
from collections import defaultdict
from typing import Dict, List, Tuple, Set


class RecommendationSystem:
    """推荐系统主类"""
    
    # 特征权重配置
    FEATURE_WEIGHTS = {
        'genre': 1.0,
        'director': 1.5,
        'actor': 1.2
    }
    
    def __init__(self, ratings: Dict[int, Dict[int, float]], 
                 movies: Dict[int, Dict]):
        """
        初始化推荐系统
        
        :param ratings: 评分数据字典 {user_id: {movie_id: rating}}
        :param movies: 电影数据字典 {movie_id: {title, genres, director, actors}}
        """
        self.ratings = ratings
        self.movies = movies
        
        # 构建物品-用户倒排索引（用于快速查找）
        self.item_users = defaultdict(set)
        for user_id, items in ratings.items():
            for movie_id in items.keys():
                self.item_users[movie_id].add(user_id)
        
        # 延迟计算的缓存
        self._item_similarity = None
        self._user_profiles = {}
    
    def _normalize_list(self, data) -> List[str]:
        """将字符串或列表统一转换为列表"""
        if isinstance(data, str):
            return data.split('|') if '|' in data else [data]
        return data if isinstance(data, list) else []
    
    def _get_movie_title(self, movie_id: int) -> str:
        """获取电影标题"""
        return self.movies.get(movie_id, {}).get('title', f'电影{movie_id}')
    
    def compute_item_similarity(self) -> Dict[int, Dict[int, float]]:
        """
        计算物品相似度矩阵（使用余弦相似度）
        
        公式：sim(A, B) = |U_A ∩ U_B| / sqrt(|U_A| × |U_B|)
        
        :return: 相似度矩阵字典 {movie_id: {similar_movie_id: similarity_score}}
        """
        if self._item_similarity is not None:
            return self._item_similarity
        
        similarity = defaultdict(dict)
        items = list(self.item_users.keys())
        
        # 计算每对物品的相似度（只计算上三角，然后对称复制）
        for i, item_a in enumerate(items):
            users_a = self.item_users[item_a]

            for item_b in items[i+1:]:
                users_b = self.item_users[item_b]
                common = len(users_a & users_b)
                
                if common > 0:
                    score = common / math.sqrt(len(users_a) * len(users_b))
                    similarity[item_a][item_b] = score
                    similarity[item_b][item_a] = score
        
        self._item_similarity = similarity
        return similarity
    
    def _generate_reason(self, movie_id: int, source_movies: List[str], 
                        method: str = "CF") -> str:
        """生成推荐理由"""
        movie_title = self._get_movie_title(movie_id)
        if method == "CF" and source_movies:
            return f"因为您喜欢《{source_movies[0]}》，所以推荐《{movie_title}》"
        elif method == "CB" and source_movies:
            features = ', '.join(source_movies[:3])
            return f"基于您的偏好（{features}等），推荐《{movie_title}》"
        return f"推荐《{movie_title}》"
    
    def item_based_recommend(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float, str]]:
        """
        基于物品的协同过滤推荐
        
        :param user_id: 用户ID
        :param top_n: 返回Top-N推荐
        :return: 推荐列表 [(movie_id, score, reason), ...]
        """
        if user_id not in self.ratings:
            return []
        
        if self._item_similarity is None:
            self.compute_item_similarity()
        
        user_rated = self.ratings[user_id]
        # [score, rated_movie_title]
        scores = defaultdict(lambda: [0.0, []])
        for rated_movie_id, rating in user_rated.items():
            for similar_movie_id, sim_score in self._item_similarity.get(rated_movie_id, {}).items():
                if similar_movie_id not in user_rated:
                    scores[similar_movie_id][0] += sim_score * rating
                    scores[similar_movie_id][1].append(self._get_movie_title(rated_movie_id))
        
        # 生成推荐列表并排序
        recommendations = [
            (movie_id, score, self._generate_reason(movie_id, reasons, "CF"))
            for movie_id, (score, reasons) in scores.items()
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]
    
    def extract_movie_features(self, movie_id: int) -> Dict[str, float]:
        """
        提取电影特征向量
        
        :param movie_id: 电影ID
        :return: 特征向量字典 {feature: weight}
        """
        movie = self.movies.get(movie_id, {})
        features = {}
        
        # 提取类型特征
        for genre in self._normalize_list(movie.get('genres', [])):
            features[f'genre_{genre}'] = self.FEATURE_WEIGHTS['genre']
        
        # 提取导演特征
        director = movie.get('director', '')
        if director:
            features[f'director_{director}'] = self.FEATURE_WEIGHTS['director']
        
        # 提取演员特征
        for actor in self._normalize_list(movie.get('actors', [])):
            features[f'actor_{actor}'] = self.FEATURE_WEIGHTS['actor']
        
        return features
    
    def build_user_profile(self, user_id: int) -> Dict[str, float]:
        """
        构建用户画像（基于历史评分的加权特征）
        
        :param user_id: 用户ID
        :return: 用户偏好向量 {feature: weight}
        """
        if user_id in self._user_profiles:
            return self._user_profiles[user_id]
        
        if user_id not in self.ratings:
            return {}
        
        profile = defaultdict(float)
        total_rating = sum(self.ratings[user_id].values())
        
        if total_rating == 0:
            return {}
        
        # 加权累加特征并归一化
        for movie_id, rating in self.ratings[user_id].items():
            for feature, weight in self.extract_movie_features(movie_id).items():
                profile[feature] += weight * rating / total_rating
        
        self._user_profiles[user_id] = dict(profile)
        return self._user_profiles[user_id]
    
    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        计算两个向量的余弦相似度
        
        :param vec1: 向量1
        :param vec2: 向量2
        :return: 相似度分数（0-1）
        """
        all_features = set(vec1.keys()) | set(vec2.keys())
        if not all_features:
            return 0.0
        
        dot_product = sum(vec1.get(f, 0) * vec2.get(f, 0) for f in all_features)
        norm1 = sum(v * v for v in vec1.values())
        norm2 = sum(v * v for v in vec2.values())
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
    
    def content_based_recommend(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float, str]]:
        """
        基于内容的推荐
        
        :param user_id: 用户ID
        :param top_n: 返回Top-N推荐
        :return: 推荐列表 [(movie_id, score, reason), ...]
        """
        if user_id not in self.ratings:
            return []
        
        user_profile = self.build_user_profile(user_id)
        if not user_profile:
            return []
        
        user_rated = set(self.ratings[user_id].keys())
        recommendations = []
        
        for movie_id, movie_info in self.movies.items():
            if movie_id in user_rated:
                continue
            
            movie_features = self.extract_movie_features(movie_id)
            similarity = self.cosine_similarity(user_profile, movie_features)
            
            if similarity > 0:
                # 提取共同特征名称
                common_features = [
                    f.split('_', 1)[1] if '_' in f else f
                    for f in user_profile.keys() if f in movie_features
                ]
                reason = self._generate_reason(movie_id, common_features, "CB")
                recommendations.append((movie_id, similarity, reason))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]
    
    def _normalize_scores(self, recommendations: List[Tuple[int, float, str]]) -> Dict[int, Dict]:
        """归一化推荐分数到[0,1]区间"""
        if not recommendations:
            return {}
        
        scores = [score for _, score, _ in recommendations]
        min_score, max_score = min(scores), max(scores)

        if max_score == min_score:
            return {
                movie_id: {
                    'score': 1.0,
                    'reason': reason
                }
                for movie_id, _, reason in recommendations
            }

        return {
            movie_id: {
                'score': (score - min_score) / (max_score - min_score),
                'reason': reason
            }
            for movie_id, score, reason in recommendations
        }
    
    def hybrid_recommend(self, user_id: int, top_n: int = 10, 
                        cf_weight: float = 0.6) -> List[Tuple[int, float, str]]:
        """
        混合推荐（协同过滤 + 基于内容）
        
        :param user_id: 用户ID
        :param top_n: 返回Top-N推荐
        :param cf_weight: 协同过滤权重（0-1）
        :return: 推荐列表 [(movie_id, score, reason), ...]
        """
        cf_recs = self.item_based_recommend(user_id, top_n * 2)
        cb_recs = self.content_based_recommend(user_id, top_n * 2)
        
        cf_scores = self._normalize_scores(cf_recs)
        cb_scores = self._normalize_scores(cb_recs)
        cb_weight = 1 - cf_weight
        
        # 融合分数
        hybrid_scores = defaultdict(float)
        reasons = {}
        
        for movie_id, data in cf_scores.items():
            hybrid_scores[movie_id] += cf_weight * data['score']
            reasons[movie_id] = data['reason']
        
        for movie_id, data in cb_scores.items():
            hybrid_scores[movie_id] += cb_weight * data['score']
            if movie_id not in reasons:
                reasons[movie_id] = data['reason']
        
        # 生成推荐列表并排序
        recommendations = [
            (mid, score, reasons.get(mid, "混合推荐"))
            for mid, score in hybrid_scores.items()
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]
    
    def print_recommendations(self, recommendations: List[Tuple[int, float, str]], 
                             title: str = "推荐结果"):
        """格式化打印推荐结果"""
        print(f"\n{'='*60}\n{title}\n{'='*60}")
        
        if not recommendations:
            print("暂无推荐")
            return
        
        for i, (movie_id, score, reason) in enumerate(recommendations, 1):
            print(f"\n{i}. {self._get_movie_title(movie_id)} (ID: {movie_id})")
            print(f"   推荐分数: {score:.4f}")
            print(f"   推荐理由: {reason}")


def create_sample_data():
    """创建示例数据"""
    # 评分数据
    ratings = {
        1: {101: 5, 102: 4, 103: 3},
        2: {101: 4, 104: 5, 105: 4},
        3: {102: 5, 103: 4, 106: 5},
        4: {101: 5, 103: 4, 107: 3},
        5: {102: 4, 104: 5, 108: 4},
        6: {101: 5, 102: 5, 109: 4}
    }
    
    # 电影数据
    movies = {
        101: {
            'title': '星际穿越',
            'genres': ['科幻', '冒险'],
            'director': '诺兰',
            'actors': ['马修·麦康纳', '安妮·海瑟薇']
        },
        102: {
            'title': '盗梦空间',
            'genres': ['科幻', '动作', '悬疑'],
            'director': '诺兰',
            'actors': ['莱昂纳多', '玛丽昂·歌迪亚']
        },
        103: {
            'title': '泰坦尼克号',
            'genres': ['爱情', '剧情'],
            'director': '卡梅隆',
            'actors': ['莱昂纳多', '凯特·温斯莱特']
        },
        104: {
            'title': '阿凡达',
            'genres': ['科幻', '冒险', '动作'],
            'director': '卡梅隆',
            'actors': ['萨姆·沃辛顿', '佐伊·索尔达娜']
        },
        105: {
            'title': '终结者2',
            'genres': ['科幻', '动作'],
            'director': '卡梅隆',
            'actors': ['阿诺·施瓦辛格', '琳达·汉密尔顿']
        },
        106: {
            'title': '禁闭岛',
            'genres': ['悬疑', '惊悚', '剧情'],
            'director': '马丁·斯科塞斯',
            'actors': ['莱昂纳多', '马克·鲁法洛']
        },
        107: {
            'title': '罗密欧与朱丽叶',
            'genres': ['爱情', '剧情'],
            'director': '巴兹·鲁赫曼',
            'actors': ['莱昂纳多', '克莱尔·丹妮丝']
        },
        108: {
            'title': '异形',
            'genres': ['科幻', '恐怖'],
            'director': '雷德利·斯科特',
            'actors': ['西格妮·韦弗']
        },
        109: {
            'title': '蝙蝠侠：黑暗骑士',
            'genres': ['动作', '犯罪', '剧情'],
            'director': '诺兰',
            'actors': ['克里斯蒂安·贝尔', '希斯·莱杰']
        }
    }
    
    return ratings, movies


def main():
    """主函数：演示推荐系统使用"""
    print("="*60)
    print("推荐系统演示")
    print("="*60)
    
    # 创建示例数据
    ratings, movies = create_sample_data()
    
    # 初始化推荐系统
    rs = RecommendationSystem(ratings, movies)
    
    # 测试用户1的推荐
    print("\n" + "="*60)
    print("测试用户1的推荐")
    print("="*60)
    print("\n用户1的评分记录：")
    for movie_id, rating in ratings[1].items():
        print(f"  - {movies[movie_id]['title']}: {rating}分")
    
    # 物品协同过滤推荐
    cf_recs = rs.item_based_recommend(1, top_n=3)
    rs.print_recommendations(cf_recs, "物品协同过滤推荐（Top-3）")
    
    # 基于内容的推荐
    cb_recs = rs.content_based_recommend(1, top_n=3)
    rs.print_recommendations(cb_recs, "基于内容的推荐（Top-3）")
    
    # 混合推荐
    hybrid_recs = rs.hybrid_recommend(1, top_n=3, cf_weight=0.6)
    rs.print_recommendations(hybrid_recs, "混合推荐（Top-3，CF权重0.6）")
    
    # 测试用户2的推荐
    print("\n\n" + "="*60)
    print("测试用户2的推荐")
    print("="*60)
    print("\n用户2的评分记录：")
    for movie_id, rating in ratings[2].items():
        print(f"  - {movies[movie_id]['title']}: {rating}分")
    
    # 物品协同过滤推荐
    cf_recs = rs.item_based_recommend(2, top_n=3)
    rs.print_recommendations(cf_recs, "物品协同过滤推荐（Top-3）")
    
    # 基于内容的推荐
    cb_recs = rs.content_based_recommend(2, top_n=3)
    rs.print_recommendations(cb_recs, "基于内容的推荐（Top-3）")
    
    # 混合推荐
    hybrid_recs = rs.hybrid_recommend(2, top_n=3, cf_weight=0.6)
    rs.print_recommendations(hybrid_recs, "混合推荐（Top-3，CF权重0.6）")


if __name__ == "__main__":
    main()


