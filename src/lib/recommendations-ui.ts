import type { Locale } from '@/i18n';

interface RecommendationsUiCopy {
  sectionTitle: string;
  shelfEyebrow: string;
  shelfTitle: string;
  shelfDescription: string;
  shelfDescriptionPrefix?: string;
  shelfDescriptionLinkLabel?: string;
  shelfDescriptionLinkPrefix?: string;
  shelfDescriptionLinkSuffix?: string;
  shelfDescriptionUrl?: string;
  detailEyebrow: string;
  detailDescriptionLabel?: string;
  sourceTitleLabel: string;
  sourcePublicationLabel: string;
  sourceLanguageLabel: string;
  sourceLinkAction: string;
}

const enCopy: RecommendationsUiCopy = {
  sectionTitle: 'Latest Recommendations',
  shelfEyebrow: 'Reading desk',
  shelfTitle: 'A hand-picked stream of worth-reading links',
  shelfDescription:
    'This section gathers outside writing, essays, and references that Aither wants to hand to you directly. When a localized version is not ready yet, some recommendations fall back to English.',
  detailEyebrow: 'Source context',
  detailDescriptionLabel: 'Why it matters',
  sourceTitleLabel: 'Source title',
  sourcePublicationLabel: 'Publication',
  sourceLanguageLabel: 'Source language',
  sourceLinkAction: 'Open source',
};

const zhHansCopy: RecommendationsUiCopy = {
  sectionTitle: '最新推荐',
  shelfEyebrow: '推荐频道',
  shelfTitle: '本频道分享优质好文、视频、播客和报告',
  shelfDescription: '致敬《湾区日报》',
  shelfDescriptionPrefix: '致敬',
  shelfDescriptionLinkPrefix: '《',
  shelfDescriptionLinkLabel: '湾区日报',
  shelfDescriptionLinkSuffix: '》',
  shelfDescriptionUrl: 'https://www.wanqu.co',
  detailEyebrow: '来源信息',
  detailDescriptionLabel: '推荐理由',
  sourceTitleLabel: '原文标题',
  sourcePublicationLabel: '来源',
  sourceLanguageLabel: '原文语言',
  sourceLinkAction: '打开原文',
};

const zhHantCopy: RecommendationsUiCopy = {
  sectionTitle: '最新推薦',
  shelfEyebrow: '推薦頻道',
  shelfTitle: '本頻道分享優質好文、影片、播客和報告',
  shelfDescription: '致敬《灣區日報》',
  shelfDescriptionPrefix: '致敬',
  shelfDescriptionLinkPrefix: '《',
  shelfDescriptionLinkLabel: '灣區日報',
  shelfDescriptionLinkSuffix: '》',
  shelfDescriptionUrl: 'https://www.wanqu.co',
  detailEyebrow: '來源資訊',
  detailDescriptionLabel: '推薦理由',
  sourceTitleLabel: '原文標題',
  sourcePublicationLabel: '來源',
  sourceLanguageLabel: '原文語言',
  sourceLinkAction: '打開原文',
};

export function getRecommendationsUiCopy(locale: Locale): RecommendationsUiCopy {
  if (locale === 'zh-CN') return zhHansCopy;
  if (locale === 'zh-TW') return zhHantCopy;
  return enCopy;
}
