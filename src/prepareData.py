from utils import get_data, save_data

df = get_data()
save_data(df, "commonData.csv")
print("✅ data prepared")