import type { Locale } from '@/i18n';

interface TranslationUiCopy {
  sectionTitle: string;
  shelfEyebrow: string;
  shelfTitle: string;
  shelfDescription: string;
  detailEyebrow: string;
  contentLanguageLabel: string;
  contentLanguageValue: string;
  originalLanguageLabel: string;
  originalTitleLabel: string;
  originalLinkLabel: string;
  originalLinkAction: string;
  translatorLabel: string;
}

const enCopy: TranslationUiCopy = {
  sectionTitle: 'Latest Translations',
  shelfEyebrow: 'Translation shelf',
  shelfTitle: 'Published in Simplified Chinese',
  shelfDescription:
    'Every translation in this section is published in Simplified Chinese. Switching the site language changes the interface only, not the translation itself.',
  detailEyebrow: 'Translation context',
  contentLanguageLabel: 'Content language',
  contentLanguageValue: 'Simplified Chinese',
  originalLanguageLabel: 'Source language',
  originalTitleLabel: 'Original title',
  originalLinkLabel: 'Original link',
  originalLinkAction: 'Read original',
  translatorLabel: 'Translator',
};

const zhHansCopy: TranslationUiCopy = {
  sectionTitle: '最新翻译',
  shelfEyebrow: '翻译频道',
  shelfTitle: '本频道发布简体中文译文',
  shelfDescription:
    '本频道译文正文统一以简体中文发布，切换站点语言只会改变界面，不会改变译文正文。',
  detailEyebrow: '译文信息',
  contentLanguageLabel: '正文语言',
  contentLanguageValue: '简体中文',
  originalLanguageLabel: '原文语言',
  originalTitleLabel: '原文标题',
  originalLinkLabel: '原文链接',
  originalLinkAction: '查看原文',
  translatorLabel: '译者',
};

const zhHantCopy: TranslationUiCopy = {
  sectionTitle: '最新翻譯',
  shelfEyebrow: '翻譯頻道',
  shelfTitle: '本頻道發布簡體中文譯文',
  shelfDescription:
    '本頻道譯文正文統一以簡體中文發布，切換站點語言只會改變介面，不會改變譯文正文。',
  detailEyebrow: '譯文資訊',
  contentLanguageLabel: '正文語言',
  contentLanguageValue: '簡體中文',
  originalLanguageLabel: '原文語言',
  originalTitleLabel: '原文標題',
  originalLinkLabel: '原文連結',
  originalLinkAction: '查看原文',
  translatorLabel: '譯者',
};

export function getTranslationsUiCopy(locale: Locale): TranslationUiCopy {
  if (locale === 'zh-CN') return zhHansCopy;
  if (locale === 'zh-TW') return zhHantCopy;
  return enCopy;
}
