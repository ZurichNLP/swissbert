from pathlib import Path
from typing import List, Tuple

import jsonlines


def create_split(articles: List["swissdox.SwissdoxArticle"]) \
        -> Tuple[List["swissdox.SwissdoxArticle"], List["swissdox.SwissdoxArticle"]]:
    """
    For each language, hold out at least 200 articles for validation.
    Use most recent articles for validation; cut between days
    """
    HOLD_OUT_SIZE = 200
    assert len(articles) > 2 * HOLD_OUT_SIZE
    articles.sort(key=lambda a: a.pubdate)

    valid = articles[-HOLD_OUT_SIZE:]
    articles = articles[:-HOLD_OUT_SIZE]
    earliest_valid_date = valid[0].pubdate
    while articles[-1].pubdate == earliest_valid_date:
        valid.append(articles.pop())

    train = articles

    assert train and valid
    return train, valid


def normalize_title(title: str) -> str:
    """
    Remove non-ASCII characters and lowercase
    """
    title = title.strip()
    title = "".join(c for c in title if ord(c) < 128)
    title = title.lower()
    return title


def extract_cheese_articles_to_exclude() -> Tuple[List[str], List[str]]:
    """
    Return titles of NZZ and Blick articles that should be excluded from pre-training data
    """
    cheese_path = Path(__file__).parent / "data_exclude/cheese.json"
    assert cheese_path.exists()
    with jsonlines.open(cheese_path) as f:
        cheese_articles = list(f)
    nzz_titles = []
    blick_titles = []
    for article in cheese_articles:
        title = normalize_title(article["title"])
        if article["source"] == "NZZ":
            nzz_titles.append(title)
        elif article["source"] == "Blick":
            blick_titles.append(title)
    assert len(nzz_titles) + len(blick_titles) == 1970
    return nzz_titles, blick_titles
