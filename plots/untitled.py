# tensor shape -> [timestamp, channels,model_level,latitude,longitude]
def mean_per_channel(,meta_data,data_1:torch.tensor,data_2:torch.tensor=None,dates=meta_data_fore["timestamps"]): 
    for i in range(data_fore_with_zeros.shape[1]):
        plt.figure(figsize=(8, 4))

        plt.plot(dates, data_1[:, i].mean((1,2,3)), label='list_fore')
        if data_2!=None:
            plt.plot(dates, data_2[:, i].mean((1,2,3)), label='list_rea')

        plt.title(f'Graph {meta_data["vars"][i]}')
        plt.xlabel('Date')
        plt.ylabel('Mean')
        plt.legend()
        plt.grid(True)

        # Set x-ticks to every six months
        six_months = timedelta(days=30*12)
        start_date = dates[0]
        end_date = dates[-1]
        ticks = [start_date + six_months * i for i in range(int((end_date - start_date) / six_months) + 1)]
        plt.xticks(ticks, [tick.strftime('%Y-%m') for tick in ticks], rotation=45)


        plt.tight_layout()  
        plt.show()
        
#________________________________________________________________________________________________________________________________________
# tensor shape -> [timestamp, channels,model_level,latitude,longitude]
def std_per_channel(meta_data,data_1:torch.tensor,data_2:torch.tensor=None,dates=meta_data_fore["timestamps"]): 
    for i in range(data_fore_with_zeros.shape[1]):
        plt.figure(figsize=(8, 4))

        plt.plot(dates, data_1[:, i].std((1,2,3)), label='list_fore')
        if data_2!=None:
            plt.plot(dates, data_2[:, i].std((1,2,3)), label='list_rea')

        plt.title(f'Graph {meta_data["vars"][i]}')
        plt.xlabel('Date')
        plt.ylabel('Mean')
        plt.legend()
        plt.grid(True)

        # Set x-ticks to every six months
        six_months = timedelta(days=30*12)
        start_date = dates[0]
        end_date = dates[-1]
        ticks = [start_date + six_months * i for i in range(int((end_date - start_date) / six_months) + 1)]
        plt.xticks(ticks, [tick.strftime('%Y-%m') for tick in ticks], rotation=45)


        plt.tight_layout()  
        plt.show()
        
#________________________________________________________________________________________________________________________________________
def reanalysis_forecast_plots(meta_data, data_1: torch.tensor, data_2: torch.tensor = None):
    import matplotlib.gridspec as gridspec

    for var in range(len(meta_data["vars"])):
        # Plot data_1
        fig = plt.figure(figsize=(20, 4))  # Adjust the figure size if needed
        gs = gridspec.GridSpec(1, 5, wspace=0.3)  # Adjust wspace to control the distance between plots
        
        for day in range(5):
            ax = fig.add_subplot(gs[day])
            x = data_1[day, var, 0].numpy()  # Convert to numpy for plotting
            ax.imshow(x)
            ax.set_title(f"data1 - {meta_data['vars'][var][0]} Day {day + 1}")
            plt.colorbar(ax.imshow(x), ax=ax, fraction=0.046, pad=0.1)
        
        fig.subplots_adjust(wspace=0.3)
        plt.show()

        # Plot data_2 if provided
        if data_2 is not None:
            fig = plt.figure(figsize=(20, 4))  # Adjust the figure size if needed
            gs = gridspec.GridSpec(1, 5, wspace=0.3)  # Adjust wspace to control the distance between plots
            
            for day in range(5):
                ax = fig.add_subplot(gs[day])
                x = data_2[day, var, 0].numpy()  # Convert to numpy for plotting
                ax.imshow(x)
                ax.set_title(f"data2 - {meta_data['vars'][var][0]} Day {day + 1}")
                plt.colorbar(ax.imshow(x), ax=ax, fraction=0.046, pad=0.1)
            
            fig.subplots_adjust(wspace=0.3)
            plt.show()
