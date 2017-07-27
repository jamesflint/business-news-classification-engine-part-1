import glob
import json
import os
import argparse
import time
from xml.etree import ElementTree as ET

import pandas as pd


def xml_converter(sourcedir, targetdir):
    """Load all xml batch files in sourcedir into memory  one by one, convert to json format, and save in targetdir"""
    start_time = time.time()
    filenumber = 0
    dirPath = ("./data/" + sourcedir + "/")

    for file in glob.iglob(os.path.join(dirPath, "*.xml")):
        with open(file) as f:
            article_tree = ET.parse(f)
            root = article_tree.getroot()

            # Parse xml for relevant fields from a single article and insert into a pandas series
            for article in root.iter("article"):
                filenumber = filenumber + 1 
                filename = str(filenumber)
                title = article.find("title").text
                content = article.find("content").text
                date = article.find("publishedDate").text
                duplicateGroupId = article.find("duplicateGroupId").text
                source_name = article.find(".//source/name").text
                source_url = article.find(".//source/homeUrl").text
                country = article.find(".//source/location/countryCode").text
                editorialRank = article.find(".//source/editorialRank").text
                xml_content = [{"title": title},
                               {"date": date},
                               {"source_name": source_name},
                               {"source_url": source_url},
                               {"country": country},
                               {"duplicateGroupID": duplicateGroupId},
                               {"editorialRank": editorialRank},
                               {"content": content}]
                xml_df = pd.Series(xml_content)

                #Convert the pandas series to json format
                json_temp = xml_df.to_json(orient="records", force_ascii=False)
                json_export = json.dumps({"content": title + content, "date": date}, indent=4, ensure_ascii=False)

                #Save json-formatted article data as json file in database
                with open("./data/" + targetdir + "/unclassified_" + filename +".json", "w") as outfile:
                    outfile.write(json_export)

    print("Finished converting, took {}".format(time.time() - start_time))


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

    # Call the function to convert the xml data
    xml_converter(sourcefile, targetdir)