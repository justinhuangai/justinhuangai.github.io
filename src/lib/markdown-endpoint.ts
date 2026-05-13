import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { getPostsByLocale, getSlug } from '@/lib/posts';
import { resolveLocale, type Locale } from '@/i18n';

const CONTENT_DIR = resolve(process.cwd(), 'src/content/posts');

export function createMarkdownStaticPaths(locale: Locale) {
  return async () => {
    const posts = await getPostsByLocale(locale);
    return posts.map((post) => ({
      params: { slug: getSlug(post.id) },
      props: { postId: post.id },
    }));
  };
}

export function createMarkdownResponse(postId: string) {
  const [localeSegment, ...slugSegments] = postId.split('/');
  const canonicalPostId =
    localeSegment && slugSegments.length > 0
      ? `${resolveLocale(localeSegment)}/${slugSegments.join('/')}`
      : postId;
  const mdPath = resolve(CONTENT_DIR, `${canonicalPostId}.md`);
  const mdxPath = resolve(CONTENT_DIR, `${canonicalPostId}.mdx`);
  const filePath = existsSync(mdxPath) ? mdxPath : mdPath;
  const raw = readFileSync(filePath, 'utf-8');

  return new Response(raw, {
    headers: { 'Content-Type': 'text/markdown; charset=utf-8' },
  });
}
