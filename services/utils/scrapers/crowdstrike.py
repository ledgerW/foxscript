import time
from datetime import datetime
from bs4 import BeautifulSoup

from utils.scrapers.base import Scraper


class CrowdstrikeScraper(Scraper):
  blog_url = None
  source = None
  base_url = 'https://www.crowdstrike.com'

  def __init__(self):
    self.driver = self.get_selenium()


  def get_latest_post_meta(self):
    self.driver.get(self.blog_url)
    time.sleep(5)
    html = self.driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")

    soup = BeautifulSoup(html, 'lxml')

    articles = soup.find_all("div", {"class": "row category_article flex-lg-row"})

    link = articles[0].find_all('h3')[0].find_all('a')[0]['href']
    self.new_post_url = self.base_url + link
    print(self.new_post_url)

    title = articles[0].find_all('h3')[0].text
    self.new_post_title = title
    print(self.new_post_title)

    post_date = articles[0].find_all('div', {'class': 'publish_info'})[0].find('p').text
    post_date = datetime.strptime(post_date, '%B %d, %Y').astimezone()
    self.new_post_date = post_date
    print(self.new_post_date)

    author = articles[0].find_all('div', {'class': 'publish_info'})[0].find('a').text
    self.new_post_author = author
    print(self.new_post_author)

    return self.new_post_url, self.new_post_date


  def scrape_post(self, url=None):
    post_url = url or self.new_post_url
    
    self.driver.get(post_url)
    time.sleep(5)
    html = self.driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")

    soup = BeautifulSoup(html, 'lxml')

    self.new_post_content = soup.find_all("div", {"class": "blog_content"})[0].text

    if url:
      self.new_post_title = soup.find_all("article")[0].find_all("h1")[0].text

      self.new_post_date = soup.find_all('div', {'class': 'publish_info'})[0].find('p').text
      self.new_post_date = datetime.strptime(self.new_post_date, '%B %d, %Y').astimezone()

      self.new_post_author = soup.find_all('div', {'class': 'publish_info'})[0].find('a').text

      self.new_post_url = post_url

    return self.new_post_content
  

class CrowdstrikeThreatIntelScraper(CrowdstrikeScraper):
  blog_url = 'https://www.crowdstrike.com/blog/category/threat-intel-research'
  source = 'Crowdstrike-ThreatIntel' 


class CrowdstrikeFrontLineScraper(CrowdstrikeScraper):
  blog_url = 'https://www.crowdstrike.com/blog/category/from-the-front-lines'
  source = 'Crowdstrike-FrontLines' 