import pandas as pd

def sresults(resultlist, outputfile="retailerlocations.csv"):
    
    result_df = pd.DataFrame(resultlist)
    result_df.to_csv(outputfile, index=False)
    print("Results saved to", outputfile)
    print(result_df.head())