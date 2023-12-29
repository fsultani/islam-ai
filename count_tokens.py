import tiktoken

def num_tokens_from_file(filename: str, encoding_name: str) -> int:
  """Returns the number of tokens in a text file."""

  with open(filename, 'r') as file:
    text = file.read()

  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding.encode(text))

  # Format the number with commas as thousands separators
  formatted_num_tokens = f"{num_tokens:,}"  # Example: 390,867

  print(f"{filename} has \033[32m{formatted_num_tokens}\033[0m tokens tokens.")

# Call the function with the file name and encoding
num_tokens_from_file("data.txt", "cl100k_base")
