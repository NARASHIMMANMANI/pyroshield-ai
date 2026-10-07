print("=" * 60)
print("Testing DiCE")
print("=" * 60)

try:
    import dice_ml

    print()
    print("DiCE Imported Successfully!")
    print("Package:", dice_ml)
    print()

except Exception as e:
    print("Error:", e)