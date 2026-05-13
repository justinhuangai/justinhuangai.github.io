import { createLlmsResponse } from '@/lib/site-content';
import { rootLocale } from '@/lib/route-helpers';

export async function GET() {
  return createLlmsResponse(rootLocale, 'summary');
}
