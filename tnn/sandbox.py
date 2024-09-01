def main():

    items = {
        "foo": 10,
        "bar": 20,
        "baz": -2,
    }

    best_item = max(items, key=items.get)
    print(best_item)


if __name__ == "__main__":
    main()
