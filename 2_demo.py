
import os
from pyspark.sql.functions import explode, col, collect_set, array_contains, to_date, month, quarter, dayofweek, \
    collect_list, from_json, array, concat_ws, flatten, udf

from pyspark.sql.functions import col, array_contains, collect_set, collect_list, to_date, month, quarter, dayofweek
from pyspark.ml.fpm import FPGrowth, PrefixSpan
from pyspark.sql.functions import col, collect_set, size, array_contains
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_json, udf, struct
from pyspark.sql.types import *
import json


item_schema = StructType([
    StructField("id", LongType(), True)
])

purchase_schema = StructType([
    StructField("average_price", DoubleType(), True),
    StructField("category", StringType(), True),
    StructField("items", ArrayType(item_schema), True)
])
def create_spark_session():
    return SparkSession.builder.appName("UserBehaviorAnalysis").config("spark.sql.shuffle.partitions", "200").config("spark.memory.offHeap.size", "2g") .config("spark.memory.offHeap.enabled", "true").config("spark.sql.parquet.enableVectorizedReader", "false").config("spark.executor.memory", "4g").config("spark.driver.memory", "4g").config("spark.hadoop.io.native.lib.available", "false").config("spark.python.worker.reuse", "false") \
        .config("spark.python.profile", "false") \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()


from pyspark.sql.functions import col, explode, from_json, udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType, LongType


import random
from pyspark.sql.functions import col, explode, from_json, udf, to_date
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType, LongType

def load_and_flatten_data(spark, parquet_path, product_json_path):
    # 1. 读取 Parquet 文件
    df = spark.read.parquet(parquet_path)

    # 2. 定义 purchase_history JSON 结构
    purchase_schema = StructType([
        StructField("average_price", DoubleType(), True),
        StructField("category", StringType(), True),
        StructField("items", ArrayType(
            StructType([StructField("id", LongType(), True)])
        ), True)
    ])

    # 3. 解析 purchase_history 字段
    df_parsed = df.withColumn("purchase_obj", from_json("purchase_history", purchase_schema))

    # 4. 拆分 items 数组
    df_exploded = df_parsed.withColumn("item", explode("purchase_obj.items"))
    df_exploded = df_exploded.withColumn("item_id", col("item.id"))

    # 5. 读取产品 JSON 并展开
    product_schema = StructType([
        StructField("products", ArrayType(
            StructType([
                StructField("id", LongType(), True),
                StructField("category", StringType(), True),
                StructField("price", DoubleType(), True)
            ])
        ), True)
    ])
    df_products_raw = spark.read.schema(product_schema).json(product_json_path)
    df_products = df_products_raw.selectExpr("explode(products) as product") \
                                 .select(
                                     col("product.id").alias("id"),
                                     col("product.category").alias("category"),
                                     col("product.price").alias("price")
                                 )

    # 6. 合并商品信息
    df_flat = df_exploded \
        .join(df_products, df_exploded["item_id"] == df_products["id"], how="left") \
        .drop(df_products["id"]) \
        .drop("item", "purchase_obj", "purchase_history")

    # 7. 添加主类映射列（main_category）
    def map_category_to_main(category):
        mapping = {
            "电子产品": ["智能手机", "笔记本电脑", "平板电脑", "智能手表", "耳机", "音响", "相机", "摄像机", "游戏机"],
            "服装": ["上衣", "裤子", "裙子", "内衣", "鞋子", "帽子", "手套", "围巾", "外套"],
            "食品": ["零食", "饮料", "调味品", "米面", "水产", "肉类", "蛋奶", "水果", "蔬菜"],
            "家居": ["家具", "床上用品", "厨具", "卫浴用品"],
            "办公": ["文具", "办公用品"],
            "运动户外": ["健身器材", "户外装备"],
            "玩具": ["玩具", "模型", "益智玩具"],
            "母婴": ["婴儿用品", "儿童课外读物"],
            "汽车用品": ["车载电子", "汽车装饰"]
        }
        for main_cat, subcats in mapping.items():
            if category in subcats:
                return main_cat
        return "其他"
    category_udf = udf(map_category_to_main, StringType())
    df_flat = df_flat.withColumn("main_category", category_udf(col("category")))

    # 8. 添加随机 payment_method
    payment_choices = ["cash", "WeChat", "Alipay", "credit card", "other"]
    random_payment_udf = udf(lambda: random.choice(payment_choices), StringType())
    df_flat = df_flat.withColumn("payment_method", random_payment_udf())

    # 9. 添加 purchase_date（从 timestamp 中提取）
    df_flat = df_flat.withColumn("purchase_date", to_date(col("timestamp")))

    # 10. 添加退款状态 purchase_status（比例 1:1:8）
    def random_status():
        r = random.random()
        if r < 0.1:
            return "已退款"
        elif r < 0.2:
            return "部分退款"
        else:
            return "未退款"
    refund_status_udf = udf(random_status, StringType())
    df_flat = df_flat.withColumn("purchase_status", refund_status_udf())

    return df, df_exploded, df_flat



def save_output(df, file_path):
    # 创建目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 处理数组列
    for column in df.columns:
        dtype = df.schema[column].dataType
        if isinstance(dtype, ArrayType):
            if isinstance(dtype.elementType, ArrayType):  # 二维数组
                df = df.withColumn(column, concat_ws(",", flatten(col(column))))
            else:  # 一维数组
                df = df.withColumn(column, concat_ws(",", col(column)))

    # 保存
    df.toPandas().to_json(file_path, orient="records", force_ascii=False, lines=True)

def analyze_category_association(df_exploded):
    """分析商品类别关联规则（只含电子产品）"""
    basket_df = df_exploded.select("id", col("purchase_obj.category").alias("category")) \
        .filter(col("category").isNotNull()) \
        .groupBy("id") \
        .agg(collect_set("category").alias("categories"))

    basket_df.cache()
    model = FPGrowth(itemsCol="categories", minSupport=0.02, minConfidence=0.5).fit(basket_df)
    rules = model.associationRules

    filtered_rules = rules.filter(
        array_contains(col("antecedent"), "电子产品") | array_contains(col("consequent"), "电子产品")
    )

    save_output(filtered_rules, "D:/DataMining/output/category_association_rules/result.json")
    return filtered_rules


def analyze_payment_category_association(df_flat):
    # df_flat = df_flat.limit(1000)
    df_flat.printSchema()
    print('-1')
    df_flat = df_flat.withColumn("payment_method", col("payment_method").cast("string"))
    df_flat.filter(col("payment_method").isNull()).show()

    df_flat.filter(col("main_category").isNull()).show()  # 查找 main_category 为空的行
    print('-2')
    df_flat_clean = df_flat.filter(col("payment_method").isNotNull() & col("main_category").isNotNull())
    df_flat_clean.select("payment_method", "main_category").describe().show()
    print('0')
    """分析支付方式与商品主类别的关联规则"""
    df_pay_combo = df_flat.select("payment_method", "main_category") \
        .filter(col("main_category").isNotNull() & col("payment_method").isNotNull())
    pay_basket = df_pay_combo.groupBy("payment_method") \
        .agg(collect_set("main_category").alias("items"))
    pay_basket.show(10, truncate=False)
    pay_basket.select("payment_method", "items").filter(size(col("items")) == 0).show(5)
    pay_basket.cache()
    pay_basket.printSchema()
    # pay_basket.show(5, truncate=False)
    print(pay_basket.count())
    pay_basket.limit(10).show(truncate=False)

    print('1')


    model = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.6).fit(pay_basket)
    rules = model.associationRules
    save_output(rules, "D:/DataMining/output/payment_category_association_rules/result.json")
    return rules






def analyze_high_value_payment(df_flat):
    """分析高价值商品（价格 > 5000）的支付方式"""
    high_value_df = df_flat.filter((col("price") > 5000) & col("payment_method").isNotNull()) \
        .groupBy("payment_method") \
        .count() \
        .orderBy("count", ascending=False)

    save_output(high_value_df, "D:/DataMining/output/high_value_payment_method/result.json")
    return high_value_df




def analyze_seasonality_and_patterns(df_flat):
    """分析季节性购买模式与用户购买序列模式"""
    df_time = df_flat.withColumn("month", month("purchase_date")) \
        .withColumn("quarter", quarter("purchase_date")) \
        .withColumn("weekday", dayofweek("purchase_date"))

    seasonality = df_time.groupBy("quarter", "main_category").count().orderBy("quarter")

    df_sorted = df_flat.select("id", "purchase_date", "main_category") \
        .filter(col("main_category").isNotNull()) \
        .orderBy("id", "purchase_date")

    df_step = df_sorted.withColumn("step", array(col("main_category")))
    user_sequences = df_step.groupBy("id").agg(collect_list("step").alias("sequence"))

    patterns = PrefixSpan(minSupport=0.005, maxPatternLength=5).findFrequentSequentialPatterns(user_sequences)

    save_output(seasonality, "D:/DataMining/output/seasonality_analysis/result.json")
    save_output(patterns, "D:/DataMining/output/seasonal_patterns/result.json")
    return seasonality, patterns



def analyze_refund_patterns(df_flat):
    """分析退款相关商品类别的关联规则"""
    refund_df = df_flat.filter(
        col("purchase_status").isin("已退款", "部分退款") & col("main_category").isNotNull()
    )

    refund_basket = refund_df.groupBy("id") \
        .agg(collect_set("main_category").alias("items"))

    refund_basket.cache()
    model = FPGrowth(itemsCol="items", minSupport=0.005, minConfidence=0.4).fit(refund_basket)
    rules = model.associationRules

    save_output(rules, "D:/DataMining/output/refund_patterns/result.json")
    return rules

def main():
    spark = create_spark_session()
    # spark.sparkContext._jsc.hadoopConfiguration().set("hadoop.native.lib", "false")
    df, df_exploded, df_flat = load_and_flatten_data(spark, "./10G_data/*.parquet", './product_catalog.json/product_catalog.json')

    # print("=== 商品类别关联规则（包含电子产品） ===")
    # category_rules = analyze_category_association(df_exploded)
    # category_rules.show(truncate=False)

    print("=== 支付方式与商品类别的关联规则 ===")
    payment_rules = analyze_payment_category_association(df_flat)
    payment_rules.show(truncate=False)

    print("=== 高价值商品支付方式分析（价格 > 5000） ===")
    high_value_payment = analyze_high_value_payment(df_flat)
    high_value_payment.show()

    print("=== 季节性购买模式与序列模式分析 ===")
    seasonality, patterns = analyze_seasonality_and_patterns(df_flat)
    seasonality.show()
    patterns.show(truncate=False)

    print("=== 退款模式分析 ===")
    refund_rules = analyze_refund_patterns(df_flat)
    refund_rules.show(truncate=False)


main()
