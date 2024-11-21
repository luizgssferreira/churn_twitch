-- CTE to join transactions with product details for the last 21 days
WITH cte_transactions_products AS (
    SELECT 
        t1.*, 
        t2.NameProduct, 
        t2.QuantityProduct
    FROM transactions AS t1
    LEFT JOIN transactions_product AS t2
        ON t1.idTransaction = t2.idTransaction
    -- Filter transactions within the last 21 days
    WHERE t1.dtTransaction < '{date}' 
      AND t1.dtTransaction >= DATE('{date}', '-21 day')
),

-- CTE to calculate product-wise metrics (quantities, points, and percentages) per customer
cte_share AS (
    SELECT 
        idCustomer,

        -- Total quantities of specific products
        SUM(CASE WHEN NameProduct = 'ChatMessage' THEN QuantityProduct ELSE 0 END) AS qtyChatMessage,
        SUM(CASE WHEN NameProduct = 'Lista de presença' THEN QuantityProduct ELSE 0 END) AS qtyAttendanceList,
        SUM(CASE WHEN NameProduct = 'Resgatar Ponei' THEN QuantityProduct ELSE 0 END) AS qtyRedeemPony,
        SUM(CASE WHEN NameProduct = 'Troca de Pontos StreamElements' THEN QuantityProduct ELSE 0 END) AS qtyPointsExchange,
        SUM(CASE WHEN NameProduct = 'Presença Streak' THEN QuantityProduct ELSE 0 END) AS qtyStreakPresence,
        SUM(CASE WHEN NameProduct = 'Airflow Lover' THEN QuantityProduct ELSE 0 END) AS qtyAirflowLover,
        SUM(CASE WHEN NameProduct = 'R Lover' THEN QuantityProduct ELSE 0 END) AS qtyRLover,

        -- Total points for specific products
        SUM(CASE WHEN NameProduct = 'ChatMessage' THEN pointsTransaction ELSE 0 END) AS pointsChatMessage,
        SUM(CASE WHEN NameProduct = 'Lista de presença' THEN pointsTransaction ELSE 0 END) AS pointsAttendanceList,
        SUM(CASE WHEN NameProduct = 'Resgatar Ponei' THEN pointsTransaction ELSE 0 END) AS pointsRedeemPony,
        SUM(CASE WHEN NameProduct = 'Troca de Pontos StreamElements' THEN pointsTransaction ELSE 0 END) AS pointsPointsExchange,
        SUM(CASE WHEN NameProduct = 'Presença Streak' THEN pointsTransaction ELSE 0 END) AS pointsStreakPresence,
        SUM(CASE WHEN NameProduct = 'Airflow Lover' THEN pointsTransaction ELSE 0 END) AS pointsAirflowLover,
        SUM(CASE WHEN NameProduct = 'R Lover' THEN pointsTransaction ELSE 0 END) AS pointsRLover,

        -- Percentage of total quantity for each product
        1.0 * SUM(CASE WHEN NameProduct = 'ChatMessage' THEN QuantityProduct ELSE 0 END) / SUM(QuantityProduct) AS pctChatMessage,
        1.0 * SUM(CASE WHEN NameProduct = 'Lista de presença' THEN QuantityProduct ELSE 0 END) / SUM(QuantityProduct) AS pctAttendanceList,
        1.0 * SUM(CASE WHEN NameProduct = 'Resgatar Ponei' THEN QuantityProduct ELSE 0 END) / SUM(QuantityProduct) AS pctRedeemPony,
        1.0 * SUM(CASE WHEN NameProduct = 'Troca de Pontos StreamElements' THEN QuantityProduct ELSE 0 END) / SUM(QuantityProduct) AS pctPointsExchange,
        1.0 * SUM(CASE WHEN NameProduct = 'Presença Streak' THEN QuantityProduct ELSE 0 END) / SUM(QuantityProduct) AS pctStreakPresence,
        1.0 * SUM(CASE WHEN NameProduct = 'Airflow Lover' THEN QuantityProduct ELSE 0 END) / SUM(QuantityProduct) AS pctAirflowLover,
        1.0 * SUM(CASE WHEN NameProduct = 'R Lover' THEN QuantityProduct ELSE 0 END) / SUM(QuantityProduct) AS pctRLover,

        -- Average quantity of 'ChatMessage' per distinct transaction day
        1.0 * SUM(CASE WHEN NameProduct = 'ChatMessage' THEN QuantityProduct ELSE 0 END) / COUNT(DISTINCT DATE(dtTransaction)) AS avgChatLive

    FROM cte_transactions_products
    GROUP BY idCustomer
),

-- CTE to aggregate product data per customer
cte_group AS (
    SELECT 
        idCustomer,
        NameProduct,
        SUM(QuantityProduct) AS totalQuantity,
        SUM(pointsTransaction) AS totalPoints
    FROM cte_transactions_products
    GROUP BY idCustomer, NameProduct
),

-- CTE to rank products by quantity (and points as a tiebreaker) per customer
cte_rank AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY idCustomer ORDER BY totalQuantity DESC, totalPoints DESC) AS rankQuantity
    FROM cte_group
    ORDER BY idCustomer
),

-- CTE to select the product with the maximum quantity per customer
cte_product_max AS (
    SELECT 
        *
    FROM cte_rank
    WHERE rankQuantity = 1
)

-- Final query to combine share metrics with the most frequently purchased product
SELECT 
    '{date}' AS dtRef,
    t1.*,
    t2.NameProduct AS maxQuantityProduct
FROM cte_share AS t1
LEFT JOIN cte_product_max AS t2
    ON t1.idCustomer = t2.idCustomer;
