-- CTE to calculate recency, frequency, and total points for the last 21 days
WITH tb_rfv AS (
    SELECT 
        idCustomer,

        -- Calculate recency in days (difference from the most recent transaction date to the reference date)
        CAST(MIN(julianday('{date}') - julianday(dtTransaction)) AS INTEGER) + 1 AS recencyDays,

        -- Count distinct days with transactions for frequency
        COUNT(DISTINCT DATE(dtTransaction)) AS frequencyDays,

        -- Sum of positive points (points earned)
        SUM(CASE WHEN pointsTransaction > 0 THEN pointsTransaction ELSE 0 END) AS pointsValue

    FROM transactions
    -- Filter transactions within the last 21 days
    WHERE dtTransaction < '{date}' 
      AND dtTransaction >= DATE('{date}', '-21 day')
    GROUP BY idCustomer
),

-- CTE to calculate the base age (number of days since the customer's last transaction)
tb_age AS (
    SELECT 
        t1.idCustomer,
        -- Calculate the base age in days (difference from the reference date to the most recent transaction date)
        CAST(MAX(julianday('{date}') - julianday(t2.dtTransaction)) AS INTEGER) + 1 AS baseAgeDays
    FROM tb_rfv AS t1
    LEFT JOIN transactions AS t2 
        ON t1.idCustomer = t2.idCustomer
    WHERE t2.dtTransaction < '{date}'
    GROUP BY t1.idCustomer
)

-- Final query to return recency, frequency, points value, base age, and email flag
SELECT 
    t1.*,                          -- Recency, frequency, and points value from tb_rfv
    '{date}' AS dtRef,             -- Reference date
    t2.baseAgeDays,                -- Base age in days from tb_age
    t3.flEmail                     -- Email flag from the customers table
FROM tb_rfv AS t1
LEFT JOIN tb_age AS t2
    ON t1.idCustomer = t2.idCustomer
LEFT JOIN customers AS t3
    ON t1.idCustomer = t3.idCustomer;
