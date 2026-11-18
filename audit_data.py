from pathlib import Path
from urllib.parse import urlsplit
import pandas as pd

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / "data" / "clean_urls.csv")


def inspect_url(url):
    try:
        parsed = urlsplit(
            url if "://" in url else "//" + url
        )
        host = parsed.hostname or ""

        return {
            "has_www": host.startswith("www."),
            "has_path": bool(parsed.path),
            "has_query": bool(parsed.query),
        }
    except ValueError:
        return {
            "has_www": False,
            "has_path": False,
            "has_query": False,
        }


checks = pd.DataFrame(
    data["URL"].map(inspect_url).tolist()
)

checks["class"] = data["label"].map({
    0: "Phishing",
    1: "Legitimate"
})

print("Percentage of URLs with each pattern:")
summary = checks.groupby("class")[
    ["has_www", "has_path", "has_query"]
].mean() * 100

print(summary.round(2).to_string())