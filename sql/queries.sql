-- name: q01_label_distribution
-- title: Распределение классов, длина и голоса «полезно»
SELECT label,
       COUNT(*) AS n_reviews,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM reviews), 1) AS share_pct,
       ROUND(AVG(n_words), 1) AS avg_words,
       ROUND(AVG(helpful_vote), 2) AS avg_helpful_votes
FROM reviews
GROUP BY label
ORDER BY n_reviews DESC;

-- name: q02_sentiment_by_year
-- title: Тональность по годам
SELECT strftime('%Y', timestamp) AS year,
       COUNT(*) AS n_reviews,
       ROUND(100.0 * SUM(label = 'positive') / COUNT(*), 1) AS pct_positive,
       ROUND(100.0 * SUM(label = 'neutral') / COUNT(*), 1) AS pct_neutral,
       ROUND(100.0 * SUM(label = 'negative') / COUNT(*), 1) AS pct_negative
FROM reviews
GROUP BY year
ORDER BY year;

-- name: q03_sentiment_by_category
-- title: Тональность по категориям товаров (категории с ≥1000 отзывов)
SELECT p.main_category,
       COUNT(*) AS n_reviews,
       ROUND(AVG(r.rating), 2) AS avg_rating,
       ROUND(100.0 * SUM(r.label = 'positive') / COUNT(*), 1) AS pct_positive,
       ROUND(100.0 * SUM(r.label = 'negative') / COUNT(*), 1) AS pct_negative
FROM reviews r
JOIN products p ON p.parent_asin = r.parent_asin
GROUP BY p.main_category
HAVING COUNT(*) >= 1000
ORDER BY n_reviews DESC;

-- name: q04_worst_products
-- title: Товары с наибольшей долей негативных отзывов (≥30 отзывов)
SELECT p.parent_asin,
       SUBSTR(p.product_title, 1, 60) AS product_title,
       COUNT(*) AS n_reviews,
       ROUND(100.0 * SUM(r.label = 'negative') / COUNT(*), 1) AS pct_negative,
       ROUND(AVG(r.rating), 2) AS avg_rating
FROM reviews r
JOIN products p ON p.parent_asin = r.parent_asin
GROUP BY p.parent_asin, p.product_title
HAVING COUNT(*) >= 30
ORDER BY pct_negative DESC, n_reviews DESC
LIMIT 15;

-- name: q05_top_products_per_category
-- title: Топ-3 товара по числу отзывов в каждой крупной категории
WITH product_stats AS (
    SELECT p.main_category,
           p.parent_asin,
           SUBSTR(p.product_title, 1, 60) AS product_title,
           COUNT(*) AS n_reviews,
           ROUND(AVG(r.rating), 2) AS avg_rating
    FROM reviews r
    JOIN products p ON p.parent_asin = r.parent_asin
    GROUP BY p.main_category, p.parent_asin, p.product_title
),
ranked AS (
    SELECT main_category, product_title, n_reviews, avg_rating,
           RANK() OVER (PARTITION BY main_category ORDER BY n_reviews DESC) AS rnk,
           SUM(n_reviews) OVER (PARTITION BY main_category) AS category_reviews
    FROM product_stats
)
SELECT main_category, rnk, product_title, n_reviews, avg_rating
FROM ranked
WHERE rnk <= 3 AND category_reviews >= 5000
ORDER BY category_reviews DESC, main_category, rnk;

-- name: q06_yearly_trend_window
-- title: Динамика среднего рейтинга по годам (LAG, скользящее среднее, накопительная сумма)
WITH yearly AS (
    SELECT CAST(strftime('%Y', timestamp) AS INTEGER) AS year,
           COUNT(*) AS n_reviews,
           AVG(rating) AS avg_rating
    FROM reviews
    GROUP BY year
)
SELECT year,
       n_reviews,
       ROUND(avg_rating, 3) AS avg_rating,
       ROUND(avg_rating - LAG(avg_rating) OVER (ORDER BY year), 3) AS change_vs_prev_year,
       ROUND(AVG(avg_rating) OVER (ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 3) AS moving_avg_3y,
       SUM(n_reviews) OVER (ORDER BY year) AS cumulative_reviews
FROM yearly
ORDER BY year;

-- name: q07_verified_purchase
-- title: Подтверждённые и неподтверждённые покупки
SELECT CASE WHEN verified_purchase = 1 THEN 'verified' ELSE 'not verified' END AS purchase_type,
       COUNT(*) AS n_reviews,
       ROUND(AVG(rating), 2) AS avg_rating,
       ROUND(100.0 * SUM(label = 'positive') / COUNT(*), 1) AS pct_positive,
       ROUND(100.0 * SUM(label = 'negative') / COUNT(*), 1) AS pct_negative,
       ROUND(AVG(n_words), 1) AS avg_words
FROM reviews
GROUP BY purchase_type;

-- name: q08_power_users
-- title: Самые активные авторы (≥20 отзывов)
SELECT user_id,
       COUNT(*) AS n_reviews,
       COUNT(DISTINCT parent_asin) AS n_products,
       ROUND(AVG(rating), 2) AS avg_rating,
       ROUND(100.0 * SUM(label = 'negative') / COUNT(*), 1) AS pct_negative
FROM reviews
GROUP BY user_id
HAVING COUNT(*) >= 20
ORDER BY n_reviews DESC
LIMIT 20;

-- name: q09_rating_vs_listed
-- title: Рейтинг Amazon (meta) против среднего в выборке (≥50 отзывов)
SELECT p.parent_asin,
       SUBSTR(p.product_title, 1, 60) AS product_title,
       p.average_rating AS listed_rating,
       ROUND(AVG(r.rating), 2) AS sample_rating,
       COUNT(*) AS n_reviews,
       ROUND(ABS(p.average_rating - AVG(r.rating)), 2) AS abs_diff
FROM reviews r
JOIN products p ON p.parent_asin = r.parent_asin
WHERE p.average_rating IS NOT NULL
GROUP BY p.parent_asin, p.product_title, p.average_rating
HAVING COUNT(*) >= 50
ORDER BY abs_diff DESC
LIMIT 15;

-- name: q10_popularity_quartiles
-- title: Популярность товара (квартили по числу отзывов) и тональность
WITH product_stats AS (
    SELECT parent_asin,
           COUNT(*) AS n_reviews,
           AVG(rating) AS avg_rating,
           100.0 * SUM(label = 'negative') / COUNT(*) AS pct_negative
    FROM reviews
    GROUP BY parent_asin
),
bucketed AS (
    SELECT parent_asin, n_reviews, avg_rating, pct_negative,
           NTILE(4) OVER (ORDER BY n_reviews) AS popularity_quartile
    FROM product_stats
)
SELECT popularity_quartile,
       COUNT(*) AS n_products,
       MIN(n_reviews) AS min_reviews,
       MAX(n_reviews) AS max_reviews,
       ROUND(AVG(avg_rating), 2) AS mean_product_rating,
       ROUND(AVG(pct_negative), 1) AS mean_pct_negative
FROM bucketed
GROUP BY popularity_quartile
ORDER BY popularity_quartile;