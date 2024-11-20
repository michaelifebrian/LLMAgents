from PIL import Image
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import io
import time
from markdownify import markdownify as md
from selenium.webdriver.common.keys import Keys
from duckduckgo_search import DDGS
import re
import requests
from pathlib import Path
import json

HF_TOKEN = ""
imgCounter = 0
def stable_diffusion_generate_image(caption: str):
    """
    A function to create image using Stable Diffusion from caption and display it. Use long specific prompt/caption to make the image accurate.

    Args:
        caption: The prompt to generate image from.
    Returns:
        A confirmation of image displayed or not
    """
    print(f"SD called with prompt: {caption}")
    global imgCounter
    def quer(payload, API_URL):
        responsesd = requests.post(API_URL, headers=headers, json=payload)
        return responsesd.content

    try:
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        image_bytes = quer({"inputs": caption, }, API_URL)
        image = Image.open(io.BytesIO(image_bytes))
        imgCounter += 1
        image.save(f"image{imgCounter}.jpg")
        return {
            "status": f"Image generated and saved as image{imgCounter}.jpg"
        }
    except Exception as error:
        return {"status": f"Error: {error}"}


def searchengineduckduckgo(keyword: str, max_results: int = 8):
    """
    A function to search the web from a keyword. ALWAYS crosschecks the information by accessing the url, the information given from search engine is not updated.

    Args:
        keyword: The keyword to search for.
        max_results: The maximum number of results to return. Increase the number for more information.
    Returns:
        A dict list of search results.
    """
    try:
        results = DDGS().text(keyword, max_results=max_results)
        return results
    except Exception as error:
        return {"status": f"Error: {error}"}


def searchenginegoogle(keyword: str):
    """
    A function to search the web from a keyword. ALWAYS crosschecks the information by accessing the url, the information given from search engine is not updated.

    Args:
        keyword: The keyword to search for.
    Returns:
        A dict list of search results.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument('--start-maximized')
    options.add_argument('window-size=1920x1080')
    # start session
    driver = webdriver.Chrome(options=options)
    returnresult = []
    try:
        # Navigate to Google
        driver.get("https://www.google.co.id")

        # # Find the search bar and input query
        search_box = driver.find_element(By.NAME, "q")
        search_query = keyword
        search_box.send_keys(search_query + Keys.RETURN)

        # Wait for results to load
        time.sleep(3)
        # html = driver.page_source
        # Extract search result elements
        results = driver.find_elements(By.CSS_SELECTOR, "div.g")
        # Loop through and extract title, link, and description
        for result in results:
            try:
                description = result.text
            except Exception:
                description = "No description"
            try:
                title_element = result.find_element(By.TAG_NAME, "h3")
                title = title_element.text
            except Exception:
                title = "No title"
            try:
                link_element = result.find_element(By.CSS_SELECTOR, "a")
                link = link_element.get_attribute("href")
            except Exception:
                link = "No link"
            result_data = {
                "title": title,
                "href": link,
                "body": description
            }
            returnresult.append(result_data)
    finally:
        # Close the driver
        driver.quit()
    return returnresult


def searchengine(keyword: str):
    """
    A function to search the web from a keyword. Always crosschecks the information by accessing the url href, the information given from search engine is not updated.

    Args:
        keyword: The keyword to search for.
    Returns:
        A dict list of search results.
    """
    print(f"Search engine called with keyword: {keyword}")
    listofresult = []
    listofresult.extend(searchengineduckduckgo(keyword))
    listofresult.extend(searchenginegoogle(keyword))
    return listofresult


def scrapeweb(url: str):
    """
    A function to open and scrape the web from a url.

    Args:
        url: The url to open.
    Returns:
        The text of the web page.
    """
    print(f"Scrape web called with url: {url}")
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument('--start-maximized')
        options.add_argument('window-size=1920x1080')
        # start session
        browser = webdriver.Chrome(options=options)
        browser.get(url)
        time.sleep(7)
        html = browser.page_source
        browser.quit()
        txt = md(html)
        txt = re.sub('\n+', '\n', txt)
        txt = re.sub('\n +', '\n', txt)
        txt = re.sub('\n+', '\n', txt)
        txt = re.sub('\n +', '\n', txt)
        return {'url': url, 'content': txt}
    except Exception as error:
        return {"status": f"Error: {error}"}


def pythoninterpreter(code_string: str):
    """
    A function to run Python code and return its output as an image. Always use this to validate a generated python code.

    Args:
        code_string: The Python code to run.
    Returns:
        The output of the Python script.
    """
    print("Python interpreter called")
    code_string = code_string.replace("{\"code_string\": \"", "").replace("{\"code_string\":\"", "")
    code_string = json.loads(json.dumps(code_string))
    code_string = code_string.encode('utf-8').decode('unicode-escape')[:-2]
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(code_string))

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        nb_out, _ = ep.preprocess(nb)
        html_exporter = HTMLExporter()
        html_data, _ = html_exporter.from_notebook_node(nb_out)
        soup = BeautifulSoup(html_data, "html.parser")
        html_data = soup
        soup.find('div', class_='jp-Cell-inputWrapper').decompose()
        with open("output.html", "w") as html_file:
            html_file.write(str(html_data))
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument('--start-maximized')
        driver = webdriver.Chrome(options=options)
        driver.get((Path.cwd() / "output.html").as_uri())
        height = driver.execute_script('return document.documentElement.scrollHeight')
        width = driver.execute_script('return document.documentElement.scrollWidth')
        driver.set_window_size(width, height)
        time.sleep(3)
        full_body_element = driver.find_element(By.TAG_NAME, "body")
        full_body_element.screenshot("output.png")
        driver.quit()
        return {'status': f"Output saved to output.png"}
    except Exception as e:
        return {'status': f"Failed to execute the notebook: {e}"}