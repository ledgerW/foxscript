from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service
from fake_useragent import UserAgent


class Scraper():
  new_post_url: str = None
  new_post_title: str = None
  new_post_date: str = None
  new_post_author: str = None
  new_post_content: str = None


  @classmethod
  def get_selenium(self):
      ua = UserAgent()
      user_agent = ua.random
      try:
          options = webdriver.ChromeOptions()
          options.add_argument("--headless")
          options.add_argument(f'--user-agent={user_agent}')
          options.add_argument("--log-level=2")
          options.add_argument("--no-sandbox")
          options.add_argument("--disable-gpu")
          options.add_argument('--disable-cache')
          options.add_argument('--disable-image-loading')
          options.add_argument("--window-size=1280x1696")
          options.add_argument("--disable-dev-shm-usage")
          options.add_argument("--disable-dev-tools")
          options.add_argument('--disable-extensions')
          options.add_argument("--dns-prefetch-disable")
          options.add_argument("--no-zygote")
          driver = Chrome(options=options)
      except:
          options = webdriver.ChromeOptions()
          options.binary_location = '/opt/chrome/chrome'
          options.add_argument('--headless')
          options.add_argument(f'--user-agent={user_agent}')
          options.add_argument("--log-level=3")
          options.add_argument('--no-sandbox')
          options.add_argument("--disable-gpu")
          options.add_argument('--disable-cache')
          options.add_argument('--disable-image-loading')
          options.add_argument("--window-size=1280x1696")
          options.add_argument("--single-process")
          options.add_argument("--disable-dev-shm-usage")
          options.add_argument("--disable-dev-tools")
          options.add_argument('--disable-extensions')
          options.add_argument('--disable-default-apps')
          options.add_argument("--dns-prefetch-disable")
          options.add_argument('--disable-browser-side-navigation')
          options.add_argument('--ignore-certificate-errors')
          options.add_argument("--no-zygote")

          service = Service(executable_path="/opt/chromedriver")
          driver = Chrome(service=service, options=options)

      return driver


  def get_latest_post_meta(self):
    raise NotImplementedError


  def scrape_post(self):
    raise NotImplementedError