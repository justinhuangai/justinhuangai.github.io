import {
  defineAitherSiteConfig,
  type AitherSiteConfig,
  type ContentSection,
  type FooterSection,
  type SocialLink,
} from '@aither/astro/site';

export type {
  AitherSiteConfig,
  ContentSection,
  FooterLink,
  FooterSection,
  SocialIcon,
  SocialLink,
} from '@aither/astro/site';

export const siteConfig = defineAitherSiteConfig({
  name: 'Justin Huang',
  title: 'Justin Huang blog',
  description: 'Justin Huang blog',
  author: {
    name: 'Justin Huang',
    avatar: '',
  },
  url: import.meta.env.SITE || 'https://justinhuangai.github.io',
  ogImage: '/og/index.png',
  images: {
    logoLight: '/logo.svg',
    logoDark: '/logo-dark.svg',
  },
  social: [
    {
      title: 'GitHub',
      href: 'https://github.com/justinhuangai',
      icon: 'github',
    },
    {
      title: 'RSS',
      href: '/rss.xml',
      icon: 'rss',
    },
  ] satisfies SocialLink[],
  blog: {
    paginationSize: 20,
    timeZone: 'Asia/Shanghai',
  },
  analytics: {
    googleAnalyticsId: import.meta.env.PUBLIC_GA_ID || '',
  },
  crisp: {
    websiteId: '',
  },
  wechatShare: {
    enabled: import.meta.env.PUBLIC_WECHAT_SHARE_ENABLED === 'true',
    appId: import.meta.env.PUBLIC_WECHAT_APP_ID || '',
    signatureEndpoint: import.meta.env.PUBLIC_WECHAT_SIGNATURE_ENDPOINT || '',
    jsSdkUrl: 'https://res.wx.qq.com/open/js/jweixin-1.6.0.js',
  },
  ui: {
    defaultMode: 'system' as const,
    defaultStyle: 'default' as const,
    enableModeSwitch: true,
    showMoreThemesMenu: true,
  },
  giscus: {
    repo: import.meta.env.PUBLIC_GISCUS_REPO || '',
    repoId: import.meta.env.PUBLIC_GISCUS_REPO_ID || '',
    category: import.meta.env.PUBLIC_GISCUS_CATEGORY || 'Announcements',
    categoryId: import.meta.env.PUBLIC_GISCUS_CATEGORY_ID || '',
    mapping: 'pathname' as const,
    reactionsEnabled: true,
    emitMetadata: false,
    inputPosition: 'top' as const,
    themeLight: 'light',
    themeDark: 'dark',
  },
  sections: [
    { id: 'translations', labelKey: 'translations', contentLocale: 'zh-CN' },
  ] as ContentSection[],
  nav: [
    { labelKey: 'blog' as const, href: '/' },
    { labelKey: 'gallery' as const, href: '/photos' },
    { labelKey: 'directory' as const, href: '/directory' },
    { labelKey: 'about' as const, href: '/about' },
  ],
  footer: {
    copyrightYear: 'auto' as 'auto' | number,
    sections: [
      {
        title: 'Navigate',
        items: [
          { title: 'Translations', href: '/translations', labelKey: 'translations' },
          { title: 'About', href: '/about', labelKey: 'about' },
          { title: 'Photos', href: '/photos', labelKey: 'gallery' },
          { title: 'Directory', href: '/directory', labelKey: 'directory' },
        ],
      },
      {
        title: 'Subscribe',
        items: [
          { title: 'RSS Feed', href: '/rss.xml' },
        ],
      },
      {
        title: 'Source',
        items: [
          {
            title: 'GitHub',
            href: 'https://github.com/justinhuangai',
            external: true,
          },
        ],
      },
    ] satisfies FooterSection[],
  },
  photosGallery: {
    paginationSize: 20,
    ogImage: '/photos/library/2023/04/2023-04-20-001.webp',
  },
  directoryPage: {
    indexTitle: 'Directory | Justin Huang',
    siteTitle: 'Justin Huang',
    indexIcp: '',
    indexIcp2: '',
    searchEngine: '',
    searchPlaceholder: '',
    iconApi: 'https://www.google.com/s2/favicons?sz=128&domain_url={url}',
    consoleHref: '',
    consoleTitle: 'Dashboard',
    backgroundColor: '#f2f2f2',
    postchatEnable: false,
    postchatButtonText: 'Chat',
    postchatButtonHref: '',
  },
} satisfies AitherSiteConfig);
