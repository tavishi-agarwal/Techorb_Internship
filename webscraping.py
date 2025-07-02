import requests
from bs4 import BeautifulSoup

website_url = "http://books.toscrape.com/"
response = requests.get(website_url)

if response.status_code == 200:
    print("✅ Website loaded successfully.\n")
    
    soup = BeautifulSoup(response.text, "html.parser")
    all_books = soup.find_all("article", class_="product_pod")

    for book in all_books:
        book_title = book.h3.a["title"]
        book_price = book.find("p", class_="price_color").text
        print(f"📖 {book_title} — 💰 {book_price}")
else:
    print("❌ Failed to retrieve the website.")
