import { getPostsByLocale, getSlug } from '@/lib/posts';
import type { Locale } from '@/i18n';

export function createMarkdownStaticPaths(locale: Locale) {
  return async () => {
    const posts = await getPostsByLocale(locale);
    return posts.map((post) => ({
      params: { slug: getSlug(post.id) },
      props: { markdownBody: post.body ?? '' },
    }));
  };
}

export function createMarkdownResponse(markdownBody: string) {
  return new Response(markdownBody, {
    headers: { 'Content-Type': 'text/markdown; charset=utf-8' },
  });
}
