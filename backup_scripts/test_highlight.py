from Components.LanguageTasks import GetHighlight

# Sample transcription data for testing
sample_transcription = """
0 - 5: Hello, welcome to this video.
5 - 10: I'm going to talk about something interesting.
10 - 15: This part is especially engaging and meaningful.
15 - 20: It contains the core message of the presentation.
20 - 25: This would make a great highlight for a short.
25 - 30: And here's the conclusion of my talk.
"""

# Test the GetHighlight function
start, end = GetHighlight(sample_transcription)
print(f"Result: start={start}, end={end}")
