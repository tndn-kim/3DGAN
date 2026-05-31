# 3D-GAN HTML 배포

## 로컬 실행
```bash
npm install
npm run dev
```

브라우저에서 표시되는 주소로 접속합니다.

## Vercel 배포
```bash
npm install
npm run build
npx vercel --prod
```

또는 GitHub에 올린 뒤 Vercel에서 해당 repository를 Import 하면 됩니다.

## Netlify 배포
```bash
npm install
npm run build
npx netlify deploy --prod --dir=dist
```

## GitHub Pages 배포
정적 HTML만 배포하려면 `index.html`을 repository 루트에 두고 GitHub Pages를 켜면 됩니다.

1. GitHub repository 생성
2. 이 폴더의 파일들을 업로드
3. Settings > Pages
4. Source: Deploy from a branch
5. Branch: main / root 선택
6. 저장 후 발급되는 URL 접속
