import json
import time
import argparse
import os

import pandas as pd
from pandas.io.json import json_normalize


def json_converter_svm(sourcefile, targetdir):
    """Load jsons from aggregated source, split into individual article, normalise fields, 
    and save as individual files in targetdir"""
    start_time = time.time()

    if not os.path.exists("./data/SVM topics/" + targetdir):
        os.makedirs("./data/SVM topics/" + targetdir)

    # Load json as string
    articles = json.load((open("./data/" + sourcefile)))

    # Normalise (i.e. breakout) nested Sources into Title and Url fields
    norm_articles_json = json_normalize(articles, "Sources")
    sources_df = norm_articles_json[["Title", "Url"]].copy()

    # Rename columns to maintain consistency with training data
    new_sources_df = sources_df.rename(columns={"Title": "Source_name", "Url": "Source_url"})

    # Load original articles as Pandas dataframe
    articles_df = pd.read_json("./data/" + sourcefile)

    # Select the other relevant columns
    cleaned_articles_df = articles_df[["Title", "PublishedDate", "Body"]].copy()

    # Concatenate the relevant columns with the normalised sources columns
    full_articles_df = pd.concat([cleaned_articles_df, new_sources_df], axis = 1)

    # Drop rows containing NaNs
    full_articles_nn = full_articles_df.dropna(how="any")

    # Convert each row of the pandas dataframe to json format
    # and save as an individual json files in data/test_jsons

    filenumber = 0

    for idx, row in full_articles_nn.iterrows():
        # print(row)
        filenumber+=1
        filename = str(filenumber)
        # json_temp = row.to_json(force_ascii=False)
        title = row[0]
        date = row[1]
        content = row[2]
        source_name = row[3]
        source_url = row[4]
        # print(title, date)
        # print(json_temp)
        json_export = json.dumps({"content": title + content, "date": date}, indent=4, ensure_ascii=False)
        # Save json-formatted article data as json file in database
        with open("./data/SVM topics/" + targetdir + "/classified_" + filename + ".json", "w") as outfile:
            outfile.write(json_export)

    print("Finished converting, took {}".format(time.time() - start_time))

    return


if __name__ == "__main__":

    # Parser setup to get source file and target dictionary
    parser = argparse.ArgumentParser()
    parser.prog = "SOURCE_FILE_&_TARGET_DIR"
    parser.description = "Get source file and target directory."
    parser.add_argument("get_sourcefile", help="Source file")
    parser.add_argument("get_targetdir", help="Target directory")
    args = parser.parse_args()
    sourcefile = str(args.get_sourcefile)
    print("Source file =", sourcefile)
    targetdir = str(args.get_targetdir)
    print("Target directory =", targetdir)

    # Call the function to convert the source json to correct format
    json_converter_svm(sourcefile, targetdir)


