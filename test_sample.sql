SELECT *
FROM orders o, customers c
WHERE o.cust_id = c.id
AND year(o.order_date) = 2024
