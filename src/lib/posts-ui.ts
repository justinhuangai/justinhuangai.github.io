import type { Locale } from '@/i18n';

interface PostsUiCopy {
  sourceEyebrow: string;
  sourceTitleLabel: string;
  sourcePublicationLabel: string;
  sourceLinkAction: string;
}

const enCopy: PostsUiCopy = {
  sourceEyebrow: 'Source context',
  sourceTitleLabel: 'Formal title',
  sourcePublicationLabel: 'Publication',
  sourceLinkAction: 'Open source',
};

const zhCnCopy: PostsUiCopy = {
  sourceEyebrow: '来源信息',
  sourceTitleLabel: '正式标题',
  sourcePublicationLabel: '来源',
  sourceLinkAction: '查看来源',
};

const zhTwCopy: PostsUiCopy = {
  sourceEyebrow: '來源資訊',
  sourceTitleLabel: '正式標題',
  sourcePublicationLabel: '來源',
  sourceLinkAction: '查看來源',
};

export function getPostsUiCopy(locale: Locale): PostsUiCopy {
  if (locale === 'zh-CN') return zhCnCopy;
  if (locale === 'zh-TW') return zhTwCopy;
  return enCopy;
}
