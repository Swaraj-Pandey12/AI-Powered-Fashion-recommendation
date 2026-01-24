from bootstrap import bootstrap

recommender = bootstrap()

tests = [
    "red dress for women",
    "black shoes for men",
    "blue shirt casual",
]

for q in tests:
    print("Query:", q)
    results = recommender.recommend(q, None)
    print("Results:", len(results))
    for r in results[:3]:
        print("-", r["name"])
    print("------")
