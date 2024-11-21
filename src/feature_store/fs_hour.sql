-- CTE to adjust transaction times to Brazil Time (UTC-3) and extract the hour
WITH cte_transactions_hour AS (
    SELECT 
        idCustomer,
        pointsTransaction,
        -- Adjusting transaction time to Brazil Time (UTC-3) and extracting the hour
        CAST(STRFTIME('%H', DATETIME(dtTransaction, '-3 hour')) AS INTEGER) AS transactionHour
    FROM transactions
    -- Filter transactions within the last 21 days
    WHERE dtTransaction < '{date}' 
      AND dtTransaction >= DATE('{date}', '-21 day')
),

-- CTE to calculate points and transaction counts by time of day (morning, afternoon, evening)
cte_share AS (
    SELECT 
        idCustomer,

        -- Total points by time of day (absolute values)
        SUM(CASE WHEN transactionHour >= 8 AND transactionHour < 12 THEN abs(pointsTransaction) ELSE 0 END) AS pointsMorning,
        SUM(CASE WHEN transactionHour >= 12 AND transactionHour < 18 THEN abs(pointsTransaction) ELSE 0 END) AS pointsAfternoon,
        SUM(CASE WHEN transactionHour >= 18 AND transactionHour <= 23 THEN abs(pointsTransaction) ELSE 0 END) AS pointsEvening,

        -- Percentage of total points by time of day
        1.0 * SUM(CASE WHEN transactionHour >= 8 AND transactionHour < 12 THEN abs(pointsTransaction) ELSE 0 END) / SUM(abs(pointsTransaction)) AS pctPointsMorning,
        1.0 * SUM(CASE WHEN transactionHour >= 12 AND transactionHour < 18 THEN abs(pointsTransaction) ELSE 0 END) / SUM(abs(pointsTransaction)) AS pctPointsAfternoon,
        1.0 * SUM(CASE WHEN transactionHour >= 18 AND transactionHour <= 23 THEN abs(pointsTransaction) ELSE 0 END) / SUM(abs(pointsTransaction)) AS pctPointsEvening,

        -- Total number of transactions by time of day
        SUM(CASE WHEN transactionHour >= 8 AND transactionHour < 12 THEN 1 ELSE 0 END) AS transactionsMorning,
        SUM(CASE WHEN transactionHour >= 12 AND transactionHour < 18 THEN 1 ELSE 0 END) AS transactionsAfternoon,
        SUM(CASE WHEN transactionHour >= 18 AND transactionHour <= 23 THEN 1 ELSE 0 END) AS transactionsEvening,

        -- Percentage of total transactions by time of day
        1.0 * SUM(CASE WHEN transactionHour >= 8 AND transactionHour < 12 THEN 1 ELSE 0 END) / SUM(1) AS pctTransactionsMorning,
        1.0 * SUM(CASE WHEN transactionHour >= 12 AND transactionHour < 18 THEN 1 ELSE 0 END) / SUM(1) AS pctTransactionsAfternoon,
        1.0 * SUM(CASE WHEN transactionHour >= 18 AND transactionHour <= 23 THEN 1 ELSE 0 END) / SUM(1) AS pctTransactionsEvening

    FROM cte_transactions_hour
    GROUP BY idCustomer
)

-- Final SELECT to include the reference date
SELECT 
    '{date}' AS dtRef,
    *
FROM cte_share;
