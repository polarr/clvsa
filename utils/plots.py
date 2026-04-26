import matplotlib.pyplot as plt
from dataset.load_dataset import *
import json


def download_trade_year(year: int):
    local_dir = download_years_of_parquets([year])
    df_1m = load_one_ticker_from_years(local_dir, TICKER, [year])
    df_5m = resample_to_5min_regular_session(df_1m)
    df_5m.to_parquet("plot_data/APPL_2022.parquet")
    
def candle_stick_plot_one_day(dates, df):
    oc_width = 0.8
    alpha=0.5
    fig, axs = plt.subplots(2, 2, figsize=(11, 5), 
                            height_ratios=[2, 1],
                            sharex=True, sharey="row")
    ticks=range(0, 80, 10)
    axs[0, 0].set_ylabel("Price (USD)")
    axs[1, 0].set_ylabel("Volume")
    
    # Candle Stick Plot
    for i in range(2):
        day_df = df[df["trade_date"] == dates[i]]
        color = np.where(day_df["close"] > day_df["open"], "green", "red")
        day_df["bottom"] = np.where(day_df["close"] > day_df["open"], 
                                    day_df["open"], 
                                    day_df["close"])
        x_idx = np.arange(len(day_df))
        
        axs[0, i].set_xticks(ticks)
        axs[0, i].set_xticklabels(df["time"][ticks])
        axs[0, i].tick_params(axis="x", labelrotation=45)
        axs[0, i].set_title(f"OCHL for {dates[i]}")
        
        # High-Low Bars
        axs[0, i].bar(x=x_idx,
                height=abs(day_df['high']-day_df["low"]), 
                bottom=day_df["low"], 
                width=0.3, 
                color=color)
        
        # Open-Close Bar
        axs[0, i].bar(x=x_idx, 
                height=abs(day_df['open']-day_df["close"]), 
                bottom=day_df["bottom"], 
                width=oc_width, 
                color=color, 
                alpha=alpha)
        
        # Volume Plot
        axs[1, i].set_xlabel("Time in 5-Minute Intervals")
        axs[1, i].bar(x=x_idx, height=day_df["volume"], color=color)
        
    plt.tight_layout(w_pad=0.5)
    plt.savefig("figs/two_day_OHCL_plot.png")
    plt.show()
    return

""" 

"""
def extract_loss_from_history(json_path):
    
    with open(json_path, 'r') as f:
        history_json = json.load(f)
    epoch, train_loss, val_loss, train_bacc, val_bacc = [[] for i in range(5)]
    
    for i in range(13):
        entry = history_json[i]
        epoch.append(entry["epoch"]) 
        train_loss.append(entry["train"]["loss"])
        val_loss.append(entry["val"]["loss"])
        train_bacc.append(entry["train"]["balanced_accuracy"])
        val_bacc.append(entry["val"]["balanced_accuracy"])
        
    data = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_bacc": train_bacc,
        "val_bacc":  val_bacc
    }
    
    data_df = pd.DataFrame(data)
    return data_df


def plot_loss_clvsa():
    history_paths = ["plot_data/clvsa_BTC.json", "plot_data/clvsa_AAPL.json"]
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))
    
    for i in range(2):
        df = extract_loss_from_history(history_paths[i])
        axs[i].plot(df["epoch"], df["train_bacc"], label="Training BACC")
        axs[i].plot(df["epoch"], df["val_bacc"], label="Validation BACC")
        axs[i].set_xlabel("Epochs")
        axs[i].legend()
      
        
    axs[0].set_title("BTCUSDT Dataset")
    axs[1].set_title("AAPL Dataset")
    axs[0].set_ylabel("Balanced Accuracy")

    plt.savefig("figs/bacc_plot.png")
    plt.show()
    return 

if __name__ == "__main__":
    file_path = "plot_data/APPL_2022.parquet"
    if not os.path.exists(file_path):
        download_trade_year(2022)
    df = pd.read_parquet(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["time"] = df["timestamp"].dt.strftime('%H:%M')

    # candle_stick_plot_one_day(["2022-01-03", "2022-01-04"], df)
    plot_loss_clvsa()
    