import { createMarkdownStaticPaths, createMarkdownResponse } from '@/lib/markdown-endpoint';
import { rootLocale } from '@/lib/route-helpers';

export const getStaticPaths = createMarkdownStaticPaths(rootLocale);

export function GET({ props }: { props: { markdownBody: string } }) {
  return createMarkdownResponse(props.markdownBody);
}
