import {
  getLocaleBasePath,
  intlLocales,
  translateTag,
  type Locale,
} from '@/i18n';
import { getPostsByLocale, type PostEntry } from '@/lib/posts';

export interface PostTagArchive {
  slug: string;
  key: string;
  label: string;
  posts: PostEntry[];
}

export interface PostTagLink {
  tag: string;
  href: string;
}

export function slugifyTaxonomyValue(value: string): string {
  return value
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^\p{Letter}\p{Number}]+/gu, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-{2,}/g, '-')
    .toLowerCase();
}

function sortArchiveByLabel<T extends { label: string }>(items: T[], locale: Locale): T[] {
  const collator = new Intl.Collator(intlLocales[locale]);
  return items.sort((left, right) => collator.compare(left.label, right.label));
}

export async function getPostTagArchives(locale: Locale): Promise<PostTagArchive[]> {
  const posts = await getPostsByLocale(locale);
  const archives = new Map<string, PostTagArchive>();

  for (const post of posts) {
    for (const tag of post.data.tags ?? []) {
      const slug = slugifyTaxonomyValue(tag);
      const existing = archives.get(slug);

      if (existing) {
        existing.posts.push(post);
        continue;
      }

      archives.set(slug, {
        slug,
        key: tag,
        label: translateTag(tag, locale),
        posts: [post],
      });
    }
  }

  return sortArchiveByLabel([...archives.values()], locale);
}

export function buildPostTagHref(tag: string, locale: Locale): string {
  return `${getLocaleBasePath(locale)}/tags/${slugifyTaxonomyValue(tag)}/`;
}

export function buildPostTagLinks(post: PostEntry, locale: Locale): PostTagLink[] {
  return (post.data.tags ?? []).map((tag) => ({
    tag,
    href: buildPostTagHref(tag, locale),
  }));
}
