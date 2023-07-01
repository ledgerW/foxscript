import time
from datetime import datetime
from bs4 import BeautifulSoup

from utils.scrapers.base import Scraper


class EWScraper(Scraper):
  blog_url = None
  source = None
  base_url = None
  source = 'EntertainmentWeekly'


  def __init__(self):
    self.driver = self.get_selenium()


  def scrape_post(self, url=None):
    self.driver.get(url)
    time.sleep(5)
    html = self.driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
    self.driver.quit()

    soup = BeautifulSoup(html, 'lxml')

    for script in soup(["script", "style"]):
      script.extract()

    title = soup.find_all("article")[0].find_all("h1")[0].text

    paras = soup.find_all("div", {"class": "paragraph"})
    content = '\n'.join(p.text for p in paras[:-3])

    date = soup.find_all('article')[0].find_all("span", {"class": "byline__block byline__block--timestamp"})[0].text
    date = date.split(' at')[0]
    try:
      date = datetime.strptime(date, '%B %d, %Y').astimezone()
    except:
      date = date.replace('Updated ', '')
      date = datetime.strptime(date, '%B %d, %Y').astimezone()

    author = soup.find_all('article')[0].find_all("a", {"class": "author-name author-text__block elementFont__detailsLinkOnly--underlined elementFont__details--bold"})[0].text

    url = url

    results = {
      'title': title,
      'content': content,
      'date': date,
      'author': author,
      'source': self.source,
      'url': url
    }

    return results