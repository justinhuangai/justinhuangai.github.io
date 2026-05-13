import { createMarkdownStaticPaths, createMarkdownResponse } from '@/lib/markdown-endpoint';
import { rootLocale } from '@/lib/route-helpers';

export const getStaticPaths = createMarkdownStaticPaths(rootLocale);

export function GET({ props }: { props: { postId: string } }) {
  return createMarkdownResponse(props.postId);
}
