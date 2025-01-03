from flipside import Flipside


with open("api_key.txt", 'r') as file:
    api_key = file.readline().strip()
    

# Initialize `Flipside` with your API Key and API Url
flipside = Flipside(api_key, "https://api-v2.flipsidecrypto.xyz")

desired_wallets = "('0xf7b10d603907658f690da534e9b7dbc4dab3e2d6', '0xbdfa4f4492dd7b7cf211209c4791af8d52bf5c50', '0x7bfee91193d9df2ac0bfe90191d40f23c773c060', '0x3e8734ec146c981e3ed1f6b582d447dde701d90c', '0x9d17bb55b57b31329cf01aa7017948e398b277bc')"
day_interval = 5

sql = """
WITH fresh_wallet_buys AS(
SELECT 
  amount_out_usd AS buy,
  block_timestamp,
  token_out,
  trader,
  tx_hash
FROM crosschain.defi.ez_dex_swaps
WHERE trader IN """ + desired_wallets + """
AND amount_out_usd IS NOT NULL
AND blockchain = 'base'
AND block_timestamp > CURRENT_TIMESTAMP() - interval '""" + str(day_interval) + """ day'
ORDER BY 1 DESC
)

SELECT DISTINCT
  
  to_varchar(buy, '$999,999,999') AS "Amount Bought",
  symbol AS "Symbol",
  block_timestamp AS "Time",
  trader AS "Trader",
  tx_hash AS "Hash"
FROM fresh_wallet_buys
INNER JOIN crosschain.core.dim_contracts
ON address = token_out
WHERE symbol NOT LIKE '%USD%'
AND symbol NOT LIKE '%DAI%'
AND symbol NOT LIKE '%ETH'
AND symbol NOT LIKE '%BTC'
AND symbol NOT LIKE '%EUR%'
ORDER BY 1 DESC
LIMIT 150
"""

# Run the query against Flipside's query engine and await the results
query_result_set = flipside.query(sql)

# Prints out query results
print(query_result_set.rows)