import { directoryGroupsSource, type LocalizedDirectoryValue } from '@/data/directory-links';
import { siteConfig } from '@/config/site';
import { getLocalizedPath, type Locale } from '@/i18n';

export interface DirectoryLink {
  spec: {
    displayName: string;
    description: string;
    url: string;
    logo?: string;
  };
  annotations: {
    show_link_anonymous: 'true' | 'false';
    to_post_radio: 'true' | 'false';
  };
  cardId: string;
  href: string;
  external: boolean;
  displayLogo: string;
}

export interface DirectoryGroup {
  spec: {
    displayName: string;
  };
  annotations: {
    icon: string;
    show_in_directory: 'true' | 'false';
    show_group_anonymous: 'true' | 'false';
  };
  links: DirectoryLink[];
  anchorId: string;
}

interface DirectorySearchConfig {
  provider: 'pagefind';
  icon: string;
  placeholder: string;
  locale: Locale;
  emptyTitle: string;
  emptyDescription: string;
}

export interface DirectoryContent {
  pageTitle: string;
  siteTitle: string;
  backgroundColor: string;
  footerThemeAttribution: string;
  footerThemeUrl: string;
  footerIcp: string;
  footerIcp2: string;
  consoleHref: string;
  consoleTitle: string;
  postchatEnabled: boolean;
  postchatButtonText: string;
  postchatButtonHref: string;
  search: DirectorySearchConfig;
  groups: DirectoryGroup[];
}

const SEARCH_ICONS = {
  baidu: '/directory/ui/images/search/baidu.svg',
  google: '/directory/ui/images/search/google.svg',
  bing: '/directory/ui/images/search/bing.svg',
  sougou: '/directory/ui/images/search/sougou.svg',
  local: '/directory/assets/search.svg',
  postchat: '/directory/ui/images/search/postchat.svg',
} as const;

const SELFHST_ICON_BASE = 'https://cdn.jsdelivr.net/gh/selfhst/icons';

function createDirectoryAnchorId(input: string, index: number): string {
  const slug = input
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, '-')
    .replace(/^-+|-+$/g, '');

  return slug ? `directory-group-${index + 1}-${slug}` : `directory-group-${index + 1}`;
}

function createDirectoryCardId(groupInput: string, linkInput: string, groupIndex: number, linkIndex: number): string {
  const slug = `${groupInput}-${linkInput}`
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, '-')
    .replace(/^-+|-+$/g, '');

  return slug
    ? `directory-card-${groupIndex + 1}-${linkIndex + 1}-${slug}`
    : `directory-card-${groupIndex + 1}-${linkIndex + 1}`;
}

function resolveLocalizedValue(locale: Locale, value: LocalizedDirectoryValue): string {
  if (typeof value === 'string') {
    return value;
  }

  return value[locale] ?? value['en-US'] ?? value['zh-CN'] ?? value['zh-TW'] ?? '';
}

function resolveSearchConfig(locale: Locale): DirectorySearchConfig {
  const configuredPlaceholder = siteConfig.directoryPage?.searchPlaceholder?.trim();
  const defaultPlaceholder =
    locale === 'zh-TW'
      ? '搜尋導航內容...'
      : locale === 'zh-CN'
        ? '搜索导航内容...'
        : 'Search the directory...';

  return {
    provider: 'pagefind',
    icon: SEARCH_ICONS.local,
    placeholder:
      configuredPlaceholder && configuredPlaceholder !== '输入搜索内容...'
        ? configuredPlaceholder
        : defaultPlaceholder,
    locale,
    emptyTitle:
      locale === 'zh-TW'
        ? '沒有找到結果'
        : locale === 'zh-CN'
          ? '没有找到结果'
          : 'No results found',
    emptyDescription:
      locale === 'zh-TW'
        ? '試試其他關鍵字或更短的搜尋詞。'
        : locale === 'zh-CN'
          ? '试试其他关键词或更短的搜索词。'
          : 'Try a different keyword or a shorter query.',
  };
}

function resolveDirectoryHref(locale: Locale, url: string): { href: string; external: boolean } {
  if (url.startsWith('/')) {
    return {
      href: getLocalizedPath(url, locale),
      external: false,
    };
  }

  return {
    href: url || '#',
    external: true,
  };
}

function resolveSelfhstLogo(reference: string): string {
  return `${SELFHST_ICON_BASE}/svg/${reference}.svg`;
}

function resolveDirectoryLogo(url: string, logo?: string): string {
  if (logo) {
    if (logo.startsWith('selfhst:')) {
      return resolveSelfhstLogo(logo.slice('selfhst:'.length));
    }

    return logo;
  }

  const iconApi =
    siteConfig.directoryPage?.iconApi?.trim() || 'https://www.google.com/s2/favicons?sz=128&domain_url={url}';
  return iconApi.replace('{url}', url);
}

export function getDirectoryContent(locale: Locale): DirectoryContent {
  const siteTitle = siteConfig.directoryPage?.siteTitle?.trim() || siteConfig.name;
  const pageTitle =
    siteConfig.directoryPage?.indexTitle?.trim() || `${siteTitle} | ${siteConfig.name}`;

  const groups: DirectoryGroup[] = directoryGroupsSource.map((group, index) => {
    const displayName = resolveLocalizedValue(locale, group.spec.displayName);

    return {
      spec: {
        displayName,
      },
      annotations: {
        icon: group.annotations.icon,
        show_in_directory: group.annotations.show_in_directory ?? 'true',
        show_group_anonymous: group.annotations.show_group_anonymous ?? 'false',
      },
      links: group.links.map((link, linkIndex) => {
        const resolved = resolveDirectoryHref(locale, link.spec.url);
        const linkDisplayName = resolveLocalizedValue(locale, link.spec.displayName);
        const linkDescription = resolveLocalizedValue(locale, link.spec.description);
        return {
          spec: {
            displayName: linkDisplayName,
            description: linkDescription,
            url: link.spec.url,
            logo: link.spec.logo,
          },
          annotations: {
            show_link_anonymous: link.annotations?.show_link_anonymous ?? 'false',
            to_post_radio: link.annotations?.to_post_radio ?? 'false',
          },
          cardId: createDirectoryCardId(displayName, linkDisplayName, index, linkIndex),
          href: resolved.href,
          external: resolved.external,
          displayLogo: resolveDirectoryLogo(link.spec.url, link.spec.logo),
        };
      }),
      anchorId: createDirectoryAnchorId(displayName, index),
    };
  });

  return {
    pageTitle,
    siteTitle,
    backgroundColor: siteConfig.directoryPage?.backgroundColor?.trim() || '#f2f2f2',
    footerThemeAttribution: '',
    footerThemeUrl: '',
    footerIcp: siteConfig.directoryPage?.indexIcp?.trim() || '',
    footerIcp2: siteConfig.directoryPage?.indexIcp2?.trim() || '',
    consoleHref: siteConfig.directoryPage?.consoleHref?.trim() || '',
    consoleTitle: siteConfig.directoryPage?.consoleTitle?.trim() || '管理后台',
    postchatEnabled: siteConfig.directoryPage?.postchatEnable ?? false,
    postchatButtonText: siteConfig.directoryPage?.postchatButtonText?.trim() || '智能对话',
    postchatButtonHref: siteConfig.directoryPage?.postchatButtonHref?.trim() || '',
    search: resolveSearchConfig(locale),
    groups,
  };
}
