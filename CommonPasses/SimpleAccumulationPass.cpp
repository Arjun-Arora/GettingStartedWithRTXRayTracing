/**********************************************************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#  * Redistributions of code must retain the copyright notice, this list of conditions and the following disclaimer.
#  * Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT
# SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************************************************/

#include "SimpleAccumulationPass.h"

namespace {
	const char* kAccumShader = "CommonPasses\\accumulate.ps.hlsl";
};


SimpleAccumulationPass::SimpleAccumulationPass(const std::string& bufferToAccumulate,const std::string& bufferHalfToAccumulate, RenderingPipeline* pipeline)
	: ::RenderPass("Accumulation Pass", "Accumulation Options"), mpRenderingPipeline(pipeline)
{
	mAccumChannel = bufferToAccumulate;
	mHalfAccumChannel = bufferHalfToAccumulate;
}

bool SimpleAccumulationPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
	if (!pResManager) return false;

	// Stash our resource manager; ask for the texture the developer asked us to accumulate
	mpResManager = pResManager;
	mpResManager->requestTextureResource(mAccumChannel);
	mpResManager->requestTextureResource(mHalfAccumChannel);

	// Create our graphics state and accumulation shader
	mpGfxState = GraphicsState::create();
	mpAccumShader = FullscreenLaunch::create(kAccumShader);

	mpHalfGfxState = GraphicsState::create();
	mpHalfAccumShader = FullscreenLaunch::create(kAccumShader);

	// Our GUI needs less space than other passes, so shrink the GUI window.
	setGuiSize(ivec2(250, 135));

	return true;
}

void SimpleAccumulationPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
	// Reset accumulation.
	mAccumCount = 0;

	// When our renderer moves around, we want to reset accumulation
	mpScene = pScene;

	// Grab a copy of the current scene's camera matrix (if it exists)
	if (mpScene && mpScene->getActiveCamera())
		mpLastCameraMatrix = mpScene->getActiveCamera()->getViewMatrix();
}

void SimpleAccumulationPass::resize(uint32_t width, uint32_t height)
{
	// Resize internal resources
	mpLastFrame = Texture::create2D(width, height, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget);
	mpHalfLastFrame = Texture::create2D(width / 2, height / 2, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget);

	// We need a framebuffer to attach to our graphics pipe state (when running our full-screen pass)
	mpInternalFbo = ResourceManager::createFbo(width, height, ResourceFormat::RGBA32Float);
	mpHalfInternalFbo = ResourceManager::createFbo(width / 2, height / 2, ResourceFormat::RGBA32Float);

	mpGfxState->setFbo(mpInternalFbo);
	mpHalfGfxState->setFbo(mpHalfInternalFbo);

	// Whenever we resize, we'd better force accumulation to restart
	mAccumCount = 0;
}

void SimpleAccumulationPass::renderGui(Gui* pGui)
{
	// Print the name of the buffer we're accumulating from and into.  Add a blank line below that for clarity
	pGui->addText((std::string("Accumulating buffer:   ") + mAccumChannel).c_str());
	pGui->addText((std::string("Accumulating buffer:   ") + mHalfAccumChannel).c_str());

	pGui->addText("");

	// Add a toggle to enable/disable temporal accumulation.  Whenever this toggles, reset the
	//     frame count and tell the pipeline we're part of that our rendering options have changed.
    if (pGui->addCheckBox(mDoAccumulation ? "Accumulating samples temporally" : "No temporal accumulation", mDoAccumulation))
    {
		mAccumCount = 0;
        setRefreshFlag();
    }

	// Add a toggle to start/end gathering data.  Whenever this toggles, reset the gathered data index to 0
	// fix button
	if (pGui->addCheckBox(mDoGathering ? "Start gathering" : "Stopped gathering", mDoGathering))
	{
		mDataCount = 0;
		setRefreshFlag();
	}

	// Display a count of accumulated frames
	pGui->addText("");
	pGui->addText((std::string("Frames accumulated: ") + std::to_string(mAccumCount)).c_str());
}

bool SimpleAccumulationPass::hasCameraMoved()
{
	// Has our camera moved?
	return mpScene &&                      // No scene?  Then the answer is no
		mpScene->getActiveCamera() &&   // No camera in our scene?  Then the answer is no
		(mpLastCameraMatrix != mpScene->getActiveCamera()->getViewMatrix());   // Compare the current matrix with the last one
}

void SimpleAccumulationPass::execute(RenderContext* pRenderContext)
{
	// Grab the texture to accumulate
	Texture::SharedPtr inputTexture = mpResManager->getTexture(mAccumChannel);

	Texture::SharedPtr inputHalfTexture = mpResManager->getTexture(mHalfAccumChannel);

	// If our input texture is invalid, or we've been asked to skip accumulation, do nothing.
	if (!inputTexture || !mDoAccumulation || !inputHalfTexture) return;

	// If the camera in our current scene has moved, we want to reset accumulation
	if (hasCameraMoved())
	{
		mAccumCount = 0;
		mpLastCameraMatrix = mpScene->getActiveCamera()->getViewMatrix();
	}

	// Set shader parameters for our accumulation
	auto shaderVars = mpAccumShader->getVars();
	shaderVars["PerFrameCB"]["gAccumCount"] = mAccumCount++;
	shaderVars["gLastFrame"] = mpLastFrame;
	shaderVars["gCurFrame"] = inputTexture;

    // Do the accumulation
    mpAccumShader->execute(pRenderContext, mpGfxState);

	// Do the half accumulation
	auto HalfshaderVars = mpHalfAccumShader->getVars();
	//accum count incremented above
	HalfshaderVars["PerFrameCB"]["gAccumCount"] = mAccumCount;
	HalfshaderVars["gLastFrame"] = mpHalfLastFrame;
	HalfshaderVars["gCurFrame"] = inputHalfTexture;

	mpHalfAccumShader->execute(pRenderContext, mpHalfGfxState);

	// gathering
	{
		if (mDoGathering && mDataCount < mMaxDataNum)
		{

			if (!(mFrameCount % mGatherRate))
			{
				// accumulation for animated scene
				// start cam animation
				if (mAccumCount >= mTargetGroundTruthSpp)
				{
					// Gather for G buffer
					Texture::SharedPtr WorldPosition = mpResManager->getTexture("WorldPosition");
					Texture::SharedPtr WorldNormal = mpResManager->getTexture("WorldNormal");
					Texture::SharedPtr MaterialDiffuse = mpResManager->getTexture("MaterialDiffuse");
					Texture::SharedPtr MaterialSpecRough = mpResManager->getTexture("MaterialSpecRough");
					Texture::SharedPtr MaterialIoR = mpResManager->getTexture("MaterialExtraParams");
					Texture::SharedPtr Emissive = mpResManager->getTexture("Emissive");

					Bitmap::ExportFlags exportFlag = Bitmap::ExportFlags::Uncompressed;
					Bitmap::FileFormat fileFormat = Bitmap::FileFormat::ExrFile;

					WorldPosition->captureToFile(0, 0, mDataPath + "WorldPosition-" + std::to_string(mDataCount) + ".exr", fileFormat, exportFlag);
					WorldNormal->captureToFile(0, 0, mDataPath + "WorldNormal-" + std::to_string(mDataCount) + ".exr", fileFormat, exportFlag);
					MaterialDiffuse->captureToFile(0, 0, mDataPath + "MaterialDiffuse-" + std::to_string(mDataCount) + ".exr", fileFormat, exportFlag);
					MaterialSpecRough->captureToFile(0, 0, mDataPath + "MaterialSpecRough-" + std::to_string(mDataCount) + ".exr", fileFormat, exportFlag);
					MaterialIoR->captureToFile(0, 0, mDataPath + "MaterialIoR-" + std::to_string(mDataCount) + ".exr", fileFormat, exportFlag);
					//Emissive->captureToFile(0, 0, "Emissive-" + std::to_string(mDataCount) + ".exr", Bitmap::FileFormat::ExrFile);

					// Gather noisy images
					inputTexture->captureToFile(0, 0, mDataPath + "Full-" + std::to_string(mDataCount) + ".exr", fileFormat, exportFlag);
					inputHalfTexture->captureToFile(0, 0, mDataPath + "Half-" + std::to_string(mDataCount) + ".exr", fileFormat, exportFlag);

					// Gather accumulated clean image
					mpLastFrame->captureToFile(0, 0, mDataPath + "Clean-" + std::to_string(mDataCount) + ".exr", fileFormat, exportFlag);

					mDataCount++;

					mpRenderingPipeline->unFreezeCam();
				}
				// freeze cam animation
				else
				{
					mpRenderingPipeline->FreezeCam();
					mFrameCount--;
				}
			}		
		}
		mFrameCount++;
	}

	// We've accumulated our result.  Copy that back to the input/output buffer
	pRenderContext->blit(mpInternalFbo->getColorTexture(0)->getSRV(), inputTexture->getRTV());
	pRenderContext->blit(mpHalfInternalFbo->getColorTexture(0)->getSRV(), inputHalfTexture->getRTV());

	// Keep a copy for next frame (we need this to avoid reading & writing to the same resource)
	pRenderContext->blit(mpInternalFbo->getColorTexture(0)->getSRV(), mpLastFrame->getRTV());
	pRenderContext->blit(mpHalfInternalFbo->getColorTexture(0)->getSRV(), mpHalfLastFrame->getRTV());

}

void SimpleAccumulationPass::stateRefreshed()
{
	// This gets called because another pass else in the pipeline changed state.  Restart accumulation
	mAccumCount = 0;
}