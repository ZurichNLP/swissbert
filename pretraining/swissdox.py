import csv
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import nh3

import utils

SEPARATOR = ' </s> '

SKIPPED_MEDIA = {
    "srf Audio",
}


@dataclass
class SwissdoxArticle:
    id: str
    pubdate: date
    medium_name: str
    language: str
    head_xml: str
    subhead_xml: str
    content_xml: str
    head_clean: str = None
    subhead_clean: str = None
    content_clean: str = None

    def __str__(self):
        return f"{self.head_xml} ({self.medium_name}, {self.pubdate})"

    def to_txt(self,
               add_metadata: bool = True,
               metadata_use_special_tokens: bool = False,
               ) -> str:
        assert self.content_clean is not None
        s = []
        if add_metadata:
            s.append(f"{' <medium> ' if metadata_use_special_tokens else SEPARATOR}{self.medium_name}")
            s.append(f"{' <year> ' if metadata_use_special_tokens else SEPARATOR}{self.pubdate.year}")
            s.append(f"{' <month> ' if metadata_use_special_tokens else SEPARATOR}{self.pubdate.strftime('%B')}")
            s.append(SEPARATOR)

        if self.head_clean:
            s.append(self.head_clean)
            s.append(SEPARATOR)
        if self.subhead_clean:
            s.append(self.subhead_clean)
            s.append(SEPARATOR)
        if self.content_clean:
            s.append(self.content_clean)
        text = ''.join(s)
        return text.strip()


class SwissdoxData:

    def __init__(self, tsv_path: Path):
        self.tsv_path = tsv_path
        self.cleaner = SwissdoxCleaner()

    def get_articles(self) -> Iterable[SwissdoxArticle]:
        cheese_articles_to_exclude = utils.extract_cheese_articles_to_exclude()
        nzz_titles = set(cheese_articles_to_exclude[0])
        blick_titles = set(cheese_articles_to_exclude[1])

        seen_article_hashes = set()
        num_duplicates = 0
        num_filtered = 0
        with self.tsv_path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if any(row[key] is None for key in ["id", "pubtime", "medium_name", "language", "head", "subhead", "content"]):
                    logging.info(f"Skipping row with missing values: {row['id']}")
                    continue
                article = SwissdoxArticle(
                    id=row["id"].strip(),
                    pubdate=date.fromisoformat(row["pubtime"].split(" ")[0]),
                    medium_name=row["medium_name"].strip(),
                    language=row["language"].strip(),
                    head_xml=row["head"].strip(),
                    subhead_xml=row["subhead"].strip(),
                    content_xml=row["content"].strip(),
                )
                assert article.language in {"de", "fr", "it", "en", "rm"}

                if article.medium_name in SKIPPED_MEDIA:
                    continue

                article_hash = hash(article.content_xml)
                if article_hash in seen_article_hashes:
                    num_duplicates += 1
                    continue
                seen_article_hashes.add(article_hash)
                article.head_clean = self.cleaner.clean(row["head"])
                article.subhead_clean = self.cleaner.clean(row["subhead"])
                article.content_clean = self.cleaner.clean(row["content"])

                # Filter out articles from cheese dataset
                if row["medium_code"].startswith("NZZ"):
                    normalized_title = utils.normalize_title(article.head_clean)
                    if normalized_title in nzz_titles:
                        num_filtered += 1
                        continue
                if "blick" in article.medium_name.lower():
                    normalized_title = utils.normalize_title(article.head_clean)
                    if normalized_title in blick_titles:
                        num_filtered += 1
                        continue

                yield article

        print(f"Skipped {num_duplicates} duplicates")
        print(f"Skipped {num_filtered} filtered articles")


class SwissdoxCleaner:

    def __init__(self):
        self.author_line_re = re.compile(r'<au>.*</au>|<ur>.*</ur>')
        self.separators_re = re.compile(r'<zt>|</zt>|<lg>|</lg>|<ka>|</ka>')
        self.paragraph_re = re.compile(r'<p>|</p>')
        self.link_start_re = re.compile(r'<a ')
        self.link_end_re = re.compile(r'</a>')
        self.sep_placeholder_re = re.compile(r'\[SEP]')
        self.double_sep_re = re.compile(rf'{SEPARATOR}\s*{SEPARATOR}')
        self.nbsp_re = re.compile(r'&nbsp;')
        self.amp_re = re.compile(r'&amp;')
        self.quot_re = re.compile(r'&quot;')
        self.lt_re = re.compile(r'&lt;')
        self.gt_re = re.compile(r'&gt;')

    def clean(self, xml: str) -> str:
        # Remove author lines
        xml = re.sub(self.author_line_re, '', xml)
        # Replace crossheadings, boxes and legends with </s>
        # Use intermediary sep symbol to avoid interference with bleach
        xml = re.sub(self.separators_re, '[SEP]', xml)
        # Add a space around hyperlinks
        xml = re.sub(self.link_start_re, ' <a ', xml)
        xml = re.sub(self.link_end_re, '</a> ', xml)
        # Replace <p> before bleach to avoid linebreaks
        xml = re.sub(self.paragraph_re, ' ', xml)
        text = nh3.clean(xml, tags=set())
        # Resolve common HTML entities
        text = re.sub(self.nbsp_re, ' ', text)
        text = re.sub(self.amp_re, '&', text)
        text = re.sub(self.quot_re, '"', text)
        text = re.sub(self.lt_re, '<', text)
        text = re.sub(self.gt_re, '>', text)
        text = text.replace('\xa0', ' ')  # nbsp
        text = text.replace('\xad', '')  # shy
        text = re.sub(self.sep_placeholder_re, SEPARATOR, text)
        # Remove duplicate separators
        text = re.sub(self.double_sep_re, SEPARATOR, text)
        text = re.sub(self.double_sep_re, SEPARATOR, text)
        text = text.replace("\n", " ")
        # Element should not be wrapped in separators
        if text.startswith(SEPARATOR):
            text = text[len(SEPARATOR):]
        if text.endswith(SEPARATOR):
            text = text[:-len(SEPARATOR)]
        text = text.strip()
        return text
