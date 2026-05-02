-- Create user activity features
CREATE TABLE user_features AS
SELECT 
    user_id,
    COUNT(interaction_id) as total_interactions,
    AVG(rating) as avg_user_rating
FROM raw_interactions
GROUP BY user_id;

-- Create item popularity features
CREATE TABLE item_features AS
SELECT 
    product_id,
    COUNT(user_id) as popularity_score,
    AVG(rating) as avg_product_rating
FROM raw_interactions
GROUP BY product_id;
