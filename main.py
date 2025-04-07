from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import plotly.express as px


# 初始化Spark
spark = SparkSession.builder.appName("UserBehaviorAnalysis").config("spark.sql.shuffle.partitions", "200").config("spark.memory.offHeap.size", "2g") .config("spark.memory.offHeap.enabled", "true").config("spark.sql.parquet.enableVectorizedReader", "false").config("spark.executor.memory", "4g").config("spark.driver.memory", "4g").getOrCreate()

# 加载所有Parquet文件
df = spark.read.parquet("./30G_data/*.parquet")

# 快速检查
print("数据量:", df.count())
# df.printSchema()
# df.show(5, vertical=True)  # 纵向显示更清晰


# 数据质量评估函数
def assess_data_quality(df):
    # 缺失值统计
    missing_stats = df.select([count(when(isnull(c) | (col(c) == ""), c)).alias(c) for c in df.columns])

    # 异常值检测
    # 异常值检测
    outlier_stats = df.select(
        count(when((col("age") < 18) | (col("age") > 100), 1).alias("invalid_age")),
              count(when(col("credit_score") < 300, 1)).alias("low_credit"),  # 修正这里
              count(when(col("income") < 0, 1)).alias("negative_income")  # 修正这里
              )

    # 类型验证
    type_issues = df.select(
        count(when(col("is_active").cast("boolean").isNull(), "is_active")).alias("invalid_boolean"),
        count(when(to_date(col("registration_date")).isNull(), "registration_date")).alias("invalid_date")
    )


    return missing_stats, outlier_stats, type_issues


def enhanced_quality_check(df):
    from pyspark.sql.functions import lit

    # 动态获取所有列名
    columns = df.columns

    # 1. 缺失值检测（包括NULL和空字符串）
    missing_results = {}
    for col_name in columns:
        null_count = df.filter(isnull(col(col_name))).count()
        empty_count = df.filter((col(col_name) == "")).count() if isinstance(df.schema[col_name].dataType,
                                                                             StringType) else 0
        missing_results[col_name] = null_count + empty_count

    # 2. 异常值检测（带调试输出）
    outlier_results = {
        "age_outliers": df.filter((col("age") < 18) | (col("age") > 100)).count(),
        "low_credit": df.filter(col("credit_score") < 300).count(),
        "negative_income": df.filter(col("income") < 0).count()
    }

    # 3. 类型验证
    type_issues = {
        "invalid_boolean": df.filter(~col("is_active").isin([True, False])).count(),
        "invalid_date": df.filter(to_date(col("registration_date")).isNull()).count()
    }

    # 打印诊断报告
    print("=== 缺失值统计 ===")
    for k, v in missing_results.items():
        print(f"{k}: {v}")

    print("\n=== 异常值统计 ===")
    for k, v in outlier_results.items():
        print(f"{k}: {v}")

    print("\n=== 类型问题 ===")
    for k, v in type_issues.items():
        print(f"{k}: {v}")

    return missing_results, outlier_results, type_issues
def select_user(df):
    # 1. 解析购买金额
    df = df.withColumn("avg_purchase",
                       regexp_extract(col("purchase_history"), '"average_price":(\d+)', 1).cast("double"))

    # 2. 计算注册天数
    df = df.withColumn("reg_days", datediff(current_date(), to_date(col("registration_date"))))

    # 3. 简单过滤异常值
    df = df.filter((col("income") > 0))
    # 定义各维度阈值（取前20%）
    income_threshold = df.approxQuantile("income", [0.8], 0.01)[0]
    credit_threshold = df.approxQuantile("credit_score", [0.8], 0.01)[0]
    reg_days_threshold = df.approxQuantile("reg_days", [0.2], 0.01)[0]  # 注册时间短的更好

    high_value_users = df.filter(
        (col("income") >= income_threshold) &
        (col("credit_score") >= credit_threshold) &
        (col("reg_days") <= reg_days_threshold)
    )



    #可视化
    # 设置支持中文的字体（使用系统自带字体）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 简体中文黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 转换为Pandas DataFrame（自动处理中小规模数据）
    hv_pd = high_value_users.toPandas()
    #收入-信用评分散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(hv_pd["income"], hv_pd["credit_score"], s=30, alpha=0.6)
    plt.xlabel("Income")
    plt.ylabel("Credit Score")
    plt.title("High Value Users: Income vs Credit")
    plt.grid(True)
    plt.savefig("income_credit_30.png")
    plt.close()
    plt.figure(figsize=(12, 4))
    # 特征分布直方图
    # 收入分布
    plt.subplot(1, 3, 1)
    plt.hist(hv_pd["income"], bins=15, color='skyblue')
    plt.title("Income Distribution")

    # 信用分分布
    plt.subplot(1, 3, 2)
    plt.hist(hv_pd["credit_score"], bins=15, color='salmon')
    plt.title("Credit Score")

    # 注册天数分布
    plt.subplot(1, 3, 3)
    plt.hist(hv_pd["reg_days"], bins=15, color='lightgreen')
    plt.title("Registration Days")

    plt.tight_layout()
    plt.savefig("hists_30.png")
    plt.close()

    #省份分布图
    # 统计省份数据
    province_count = high_value_users.groupBy("province").count()
    province_pd = province_count.orderBy("count", ascending=False).limit(10).toPandas()

    plt.figure(figsize=(10, 5))
    plt.bar(province_pd["province"], province_pd["count"], color='orange')
    plt.xticks(rotation=45)
    plt.title("Top 10 Provinces by User Count")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig("province_bar_30.png")
    plt.close()



missing_stats, outlier_stats, type_issues = enhanced_quality_check(df)


# 解析JSON字段
purchase_schema = StructType([
    StructField("average_price", DoubleType()),
    StructField("purchase_frequency", IntegerType())
])

df_clean = (df
            .withColumn("purchase_data", from_json(col("purchase_history"), purchase_schema))
            .withColumn("avg_purchase", col("purchase_data.average_price"))
            .withColumn("purchase_freq", col("purchase_data.purchase_frequency"))

            # 处理异常值
            .withColumn("age", when((col("age") < 18) | (col("age") > 100), None).otherwise(col("age")))
            .withColumn("income", when(col("income") < 0, 0).otherwise(col("income")))

            # 提取省份
            .withColumn("province", regexp_extract(col("chinese_address"), "^(.*?省)", 1))

            # 转换日期
            .withColumn("reg_days", datediff(current_date(), to_date(col("registration_date"))))
            )

select_user(df_clean)


