url = input("URL: ").strip()
print(url)

username = url.replace("https://twitter.com/", "")
print(f"Username: {username}")


username = url.removeprefix("https://twitter.com/")
print(f"Username: {username}")
