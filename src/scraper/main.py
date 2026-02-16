import asyncio
import nest_asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import pandas as pd

nest_asyncio.apply()


async def get_book_details(browser_context, book_url):
    page = await browser_context.new_page()
    try:
        full_url = (
            f"https://www.goodreads.com{book_url}"
            if book_url.startswith("/")
            else book_url
        )
        await page.goto(full_url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(2)

        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")

        desc_element = soup.select_one(
            ".BookPageMetadataSection__description .Formatted, .DescriptionText, #description"
        )
        description = (
            desc_element.get_text(strip=True) if desc_element else "Brak opisu"
        )

        genre_elements = soup.select(
            "a[href*='/genres/'], "
            ".BookPageMetadataSection__genres a, "
            ".bookPageGenreLink, "
            ".actionLinkLite.genreLink"
        )

        genres = []
        for g in genre_elements:
            text = g.get_text(strip=True)
            if text and text.lower() not in ["...more", "edit details", "genres"]:
                genres.append(text)

        await page.close()
        return description, list(set(genres))
    except Exception:
        await page.close()
        return "Błąd pobierania", []


async def scrape_goodreads(user_url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        user_id = user_url.split("/")[-1].split("-")[0]
        shelf_url = f"https://www.goodreads.com/review/list/{user_id}?shelf=read&per_page=50&view=table"

        print(f"Otwieranie: {shelf_url}")
        await page.goto(shelf_url)

        print("--- TRYB OCZEKIWANIA ---")
        print("1. Zaloguj się, jeśli trzeba.")
        print("2. Upewnij się, że widzisz listę książek.")
        print("3. Jeśli lista jest, a skrypt nie rusza, przewiń stronę trochę w dół.")

        found_rows = None
        for i in range(120):
            try:
                rows = await page.query_selector_all(
                    "tr.book_record, .book_record, tr[id^='review_']"
                )
                if len(rows) > 0:
                    print(f"Sukces! Wykryto {len(rows)} wierszy danych.")
                    found_rows = rows
                    break

                if i % 10 == 0:
                    print(f"Czekam... (sekunda {i}). Obecny URL: {page.url}")
            except Exception:
                pass
            await asyncio.sleep(1)

        if not found_rows:
            print("BŁĄD: Nie znaleziono książek. Zamykam.")
            await browser.close()
            return []

        await asyncio.sleep(3)

        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")

        items = soup.select("tr.book_record, tr[id^='review_']")

        books_data = []
        print(f"Rozpoczynam analizę {len(items)} pozycji...")

        for row in items:
            try:
                title_link = row.select_one(".field.title a, td.field.title a")
                author_link = row.select_one(".field.author a, td.field.author a")

                if not title_link:
                    continue

                title = title_link.get_text(strip=True)
                author = (
                    author_link.get_text(strip=True).replace(" *", "")
                    if author_link
                    else "Unknown"
                )
                book_link = title_link["href"]

                print(f"Przetwarzanie: {title}")
                desc, genres = await get_book_details(context, book_link)

                books_data.append(
                    {
                        "book_id": book_link.split("/")[-1],
                        "title": title,
                        "author": author,
                        "genres": ", ".join(genres),
                        "description": desc,
                    }
                )
            except Exception as e:
                print(f"Pominięto element: {e}")

        await browser.close()
        return books_data


user_profile_url = "https://www.goodreads.com/user/show/50043037-talkincloud"
data = asyncio.run(scrape_goodreads(user_profile_url))

if data:
    df = pd.DataFrame(data)
    df.to_csv("user_history.csv", index=False, encoding="utf-8")
    print(f"\nZapisano {len(df)} książek do pliku CSV.")
else:
    print("\nNie udało się zapisać żadnych danych.")
