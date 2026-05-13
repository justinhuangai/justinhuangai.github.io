import {
  sortEntriesByPinnedDate,
  getEntrySlug,
  getLocalizedCollectionEntries,
  getLocalizedPosts,
  type AitherLocalizedEntry,
  type AitherPostEntry,
} from '@aither/astro/posts';
import { getCollection, type CollectionKey } from 'astro:content';
import {
  getSectionFallbackLocale,
  resolveSectionContentLocale,
} from '@/lib/content-sections';
import { locales, resolveLocale, type Locale } from '@/i18n';

export type LocalizedEntry<K extends CollectionKey> = AitherLocalizedEntry<K>;
export interface PostFrontmatterExtensions {
  sourceTitle?: string;
  sourceUrl?: string;
  sourcePublication?: string;
}

export type PostEntry = Omit<AitherPostEntry, 'data'> & {
  data: AitherPostEntry['data'] & PostFrontmatterExtensions;
};

export async function getPostsByLocale(locale: Locale) {
  return (await getLocalizedPosts(locale)) as PostEntry[];
}

/** Generic: fetch any content collection filtered by locale, sorted by pin + date */
export async function getContentByLocale<K extends CollectionKey>(
  collection: K,
  locale: Locale,
): Promise<LocalizedEntry<K>[]> {
  const contentLocale = resolveSectionContentLocale(collection, locale);
  const primaryEntries = await getLocalizedCollectionEntries(collection, contentLocale);
  const fallbackLocale = getSectionFallbackLocale(collection);

  if (!fallbackLocale || fallbackLocale === contentLocale) {
    return primaryEntries;
  }

  const fallbackEntries = await getLocalizedCollectionEntries(collection, fallbackLocale);
  const dedupedEntries = new Map<string, LocalizedEntry<K>>();

  for (const entry of primaryEntries) {
    dedupedEntries.set(getSlug(entry.id), entry);
  }

  for (const entry of fallbackEntries) {
    const slug = getSlug(entry.id);
    if (!dedupedEntries.has(slug)) {
      dedupedEntries.set(slug, entry);
    }
  }

  return [...dedupedEntries.values()].sort(sortEntriesByPinnedDate);
}

/** Strip the locale prefix from post ID to get the slug */
export function getSlug(postId: string): string {
  return getEntrySlug(postId);
}

export function getEntryLocale(entryId: string): Locale {
  return resolveLocale(entryId.split('/')[0] ?? 'en');
}

export async function getAvailableLocalesForSlug<K extends CollectionKey>(
  collection: K,
  slug: string,
): Promise<Locale[]> {
  const entries = await getCollection(collection);
  const localeSet = new Set<Locale>();

  entries.forEach((entry) => {
    if (getSlug(entry.id) === slug) {
      localeSet.add(getEntryLocale(entry.id));
    }
  });

  return locales.filter((locale) => localeSet.has(locale));
}
