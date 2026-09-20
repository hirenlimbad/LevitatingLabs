export function getReadingTime(content: string): { minutes: number; wordCount: number; text: string } {
  if (!content) return { minutes: 1, wordCount: 0, text: "1 min read" };

  // Strip code blocks, HTML tags, and markdown formatting for accurate word count
  const cleanText = content
    .replace(/```[\s\S]*?```/g, "")
    .replace(/<[^>]*>/g, "")
    .replace(/#+\s+/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .trim();

  const words = cleanText.split(/\s+/).filter(w => w.length > 0);
  const wordCount = words.length;
  const minutes = Math.max(1, Math.ceil(wordCount / 200));

  return {
    minutes,
    wordCount,
    text: `${minutes} min read`,
  };
}
